"""
x8_mavlink.py — MAVLink connection management.

Handles everything that touches the wire:
  - connecting to ArduPilot SITL
  - background thread that reads incoming messages into a StateBuffer
  - functions that send attitude targets and RC overrides

Nothing in here knows about the control law.
"""

import copy
import math
import threading
import time
from dataclasses import dataclass

from pymavlink import mavutil


# ---------------------------------------------------------------------------
# Vehicle state snapshot
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    # Attitude  (radians)
    phi:      float = 0.0
    theta:    float = 0.0
    psi:      float = 0.0
    # Body rates  (rad/s)
    p:        float = 0.0
    q:        float = 0.0
    r:        float = 0.0
    # Air data
    airspeed: float = 17.0
    alpha:    float = 0.0    # rad — needs AOA_ENABLE=1 in ArduPilot params
    beta:     float = 0.0    # rad
    # Vehicle status
    armed:    bool  = False
    mode:     int   = -1
    # Freshness flag
    valid:    bool  = False


class StateBuffer:
    """Thread-safe latest-value store written by rx thread, read by main loop."""

    def __init__(self):
        self._lock  = threading.Lock()
        self._state = VehicleState()

    def write(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)
            self._state.valid = True

    def read(self) -> VehicleState:
        with self._lock:
            return copy.copy(self._state)


# ---------------------------------------------------------------------------
# Receive thread
# ---------------------------------------------------------------------------

class MAVReceiver(threading.Thread):
    """
    Reads MAVLink messages in a background daemon thread.
    Writes parsed fields into StateBuffer.

    Messages consumed:
        ATTITUDE    → phi, theta, psi, p, q, r
        VFR_HUD     → airspeed
        AOA_SSA     → alpha, beta  (set AOA_ENABLE=1 in ArduPilot)
        HEARTBEAT   → armed, mode
    """

    def __init__(self, conn, buf: StateBuffer, verbose: bool = False):
        super().__init__(daemon=True, name="MAVReceiver")
        self.conn    = conn
        self.buf     = buf
        self.verbose = verbose
        self._stop   = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            msg = self.conn.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                continue
            t = msg.get_type()

            if t == 'ATTITUDE':
                self.buf.write(
                    phi=msg.roll, theta=msg.pitch, psi=msg.yaw,
                    p=msg.rollspeed, q=msg.pitchspeed, r=msg.yawspeed,
                )
            elif t == 'VFR_HUD':
                self.buf.write(airspeed=max(float(msg.airspeed), 1.0))
            elif t == 'AOA_SSA':
                self.buf.write(
                    alpha=math.radians(msg.AOA),
                    beta=math.radians(msg.SSA),
                )
            elif t == 'HEARTBEAT':
                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self.buf.write(armed=armed, mode=int(msg.custom_mode))
            elif t == 'STATUSTEXT' and self.verbose:
                print(f"[AP] {msg.text}")


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

GUIDED_MODE = 15
FBWA_MODE   = 2


def connect(address: str, stream_hz: int = 50) -> mavutil.mavfile:
    """
    Open MAVLink connection, wait for heartbeat, request attitude stream.

    address examples:
        'tcp:127.0.0.1:5762'   (SITL default)
        'udp:127.0.0.1:14550'
        '/dev/ttyUSB0,57600'   (telemetry radio)
    """
    print(f"[MAV] Connecting to {address} ...")
    conn = mavutil.mavlink_connection(address, autoreconnect=True)
    conn.wait_heartbeat(timeout=15)
    print(f"[MAV] Heartbeat from system {conn.target_system} "
          f"component {conn.target_component}")

    # Request ATTITUDE at stream_hz, VFR_HUD at 20 Hz
    conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, stream_hz, 1,
    )
    return conn


def wait_for_state(buf: StateBuffer, timeout: float = 10.0) -> bool:
    """Block until StateBuffer contains at least one valid snapshot."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if buf.read().valid:
            return True
        time.sleep(0.05)
    return False


def set_mode(conn, mode_num: int):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_num, 0, 0, 0, 0, 0,
    )


# ---------------------------------------------------------------------------
# Output senders
# ---------------------------------------------------------------------------

def send_attitude_target(conn,
                         roll_d:       float,
                         pitch_d:      float,
                         yaw_d:        float,
                         roll_rate_d:  float = 0.0,
                         pitch_rate_d: float = 0.0,
                         yaw_rate_d:   float = 0.0,
                         thrust:       float = 0.6):
    """
    Send SET_ATTITUDE_TARGET.
    ArduPlane uses the quaternion for attitude hold; rates are feedforward.
    """
    cy, sy = math.cos(yaw_d   / 2), math.sin(yaw_d   / 2)
    cp, sp = math.cos(pitch_d / 2), math.sin(pitch_d / 2)
    cr, sr = math.cos(roll_d  / 2), math.sin(roll_d  / 2)
    q = [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
    conn.mav.set_attitude_target_send(
        int(time.monotonic() * 1000) & 0xFFFFFFFF,
        conn.target_system,
        conn.target_component,
        0b00000000,
        q,
        roll_rate_d, pitch_rate_d, yaw_rate_d,
        thrust,
    )


def send_rc_override(conn,
                     delta_L_rad: float,
                     delta_R_rad: float,
                     throttle_pct: float = 60.0,
                     elevon_limit_deg: float = 30.0):
    """
    Alternative output: override RC channels directly.
    X8 default ArduPilot mixer: Ch1 = right elevon, Ch2 = left elevon.
    ±elevon_limit_deg → PWM 1000–2000 µs.
    """
    def to_pwm(rad: float) -> int:
        pct = max(-1.0, min(1.0, math.degrees(rad) / elevon_limit_deg))
        return int(1500 + pct * 400)

    conn.mav.rc_channels_override_send(
        conn.target_system, conn.target_component,
        to_pwm(delta_R_rad),            # Ch1
        to_pwm(delta_L_rad),            # Ch2
        int(1000 + throttle_pct * 10),  # Ch3 throttle
        65535, 65535, 65535, 65535, 65535,
    )
