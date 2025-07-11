"""
Central constants and configuration for DexHand environment.

This module defines all shared constants to ensure single source of truth.
"""

# DOF dimensions
NUM_BASE_DOFS = 6  # ARTx, ARTy, ARTz, ARRx, ARRy, ARRz
NUM_ACTIVE_FINGER_DOFS = 12  # 12 finger controls mapping to 19 DOFs with coupling
NUM_TOTAL_FINGER_DOFS = 20  # 5 fingers × 4 joints (including fixed joint3_1)
NUM_FINGERS = 5  # Thumb, index, middle, ring, pinky

# Joint names
BASE_JOINT_NAMES = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]

FINGER_JOINT_NAMES = [
    "r_f_joint1_1",
    "r_f_joint1_2",
    "r_f_joint1_3",
    "r_f_joint1_4",
    "r_f_joint2_1",
    "r_f_joint2_2",
    "r_f_joint2_3",
    "r_f_joint2_4",
    "r_f_joint3_1",
    "r_f_joint3_2",
    "r_f_joint3_3",
    "r_f_joint3_4",
    "r_f_joint4_1",
    "r_f_joint4_2",
    "r_f_joint4_3",
    "r_f_joint4_4",
    "r_f_joint5_1",
    "r_f_joint5_2",
    "r_f_joint5_3",
    "r_f_joint5_4",
]

# Body names for fingertips and fingerpads
FINGERTIP_BODY_NAMES = [
    "r_f_link1_tip",
    "r_f_link2_tip",
    "r_f_link3_tip",
    "r_f_link4_tip",
    "r_f_link5_tip",
]

FINGERPAD_BODY_NAMES = [
    "r_f_link1_pad",
    "r_f_link2_pad",
    "r_f_link3_pad",
    "r_f_link4_pad",
    "r_f_link5_pad",
]

# Finger DOF coupling mapping (12 actions -> 19 DOFs)
# Actions map to finger DOFs as follows:
# 0: r_f_joint1_1 (thumb spread)
# 1: r_f_joint1_2 (thumb MCP)
# 2: r_f_joint1_3, r_f_joint1_4 (thumb DIP - coupled)
# 3: r_f_joint2_1, r_f_joint4_1, r_f_joint5_1 (finger spread - coupled, 5_1 is 2x)
# 4: r_f_joint2_2 (index MCP)
# 5: r_f_joint2_3, r_f_joint2_4 (index DIP - coupled)
# 6: r_f_joint3_2 (middle MCP)
# 7: r_f_joint3_3, r_f_joint3_4 (middle DIP - coupled)
# 8: r_f_joint4_2 (ring MCP)
# 9: r_f_joint4_3, r_f_joint4_4 (ring DIP - coupled)
# 10: r_f_joint5_2 (pinky MCP)
# 11: r_f_joint5_3, r_f_joint5_4 (pinky DIP - coupled)
# Note: r_f_joint3_1 is fixed at 0 (not controlled)
FINGER_COUPLING_MAP = {
    0: ["r_f_joint1_1"],  # thumb spread
    1: ["r_f_joint1_2"],  # thumb MCP
    2: ["r_f_joint1_3", "r_f_joint1_4"],  # thumb DIP (coupled)
    3: [
        ("r_f_joint2_1", 1.0),
        ("r_f_joint4_1", 1.0),
        ("r_f_joint5_1", 2.0),
    ],  # finger spread (5_1 is 2x)
    4: ["r_f_joint2_2"],  # index MCP
    5: ["r_f_joint2_3", "r_f_joint2_4"],  # index DIP (coupled)
    6: ["r_f_joint3_2"],  # middle MCP
    7: ["r_f_joint3_3", "r_f_joint3_4"],  # middle DIP (coupled)
    8: ["r_f_joint4_2"],  # ring MCP
    9: ["r_f_joint4_3", "r_f_joint4_4"],  # ring DIP (coupled)
    10: ["r_f_joint5_2"],  # pinky MCP
    11: ["r_f_joint5_3", "r_f_joint5_4"],  # pinky DIP (coupled)
}
