"""Inidcices for GST."""

# kinematic chain link indices for the robot
I_CHAIN_LINK_23: int = 0
I_CHAIN_LINK_34: int = 1
I_CHAIN_LINK_4prime5: int = 2
I_CHAIN_LINK_56: int = 3
I_CHAIN_LINK_67: int = 4
I_CHAIN_LINK_38: int = 5

# indices for connection links
I_CONNECTOR_LINK_GST_23: int = 0
I_CONNECTOR_LINK_DFT_C5: int = 1
I_CONNECTOR_LINK_EDT1_C4: int = 2
I_CONNECTOR_LINK_EDT1_5C: int = 3
I_CONNECTOR_LINK_EDT2_C4: int = 4
I_CONNECTOR_LINK_KFT_3C: int = 5

# indices for connector offsets => custom thetas for every connector
I_CONNECTOR_OFFSET_DFT_C5: int = 0
I_CONNECTOR_OFFSET_EDT1_C4: int = 1
I_CONNECTOR_OFFSET_EDT1_5C: int = 2
I_CONNECTOR_OFFSET_EDT2_C4: int = 3
I_CONNECTOR_OFFSET_KFT_3C: int = 4


# link indices for the robot, including virtual links for tendon attachment points
I_LINK_23: int = 0
I_LINK_34: int = 1
I_LINK_4prime5: int = 2
I_LINK_56: int = 3
I_LINK_67: int = 4
I_LINK_38: int = 5
I_LINK_GST_23: int = 6
I_LINK_DFT_C5: int = 7
I_LINK_EDT1_C4: int = 8
I_LINK_EDT1_5C: int = 9
I_LINK_EDT2_C4: int = 10
I_LINK_KFT_3C: int = 11

# joint indices for the robot
I_JOINT_3: int = 0
I_JOINT_4: int = 1
I_JOINT_5: int = 2
I_JOINT_6: int = 3
I_JOINT_8: int = 4

# indices for pulley radii
I_RADIUS_GST_3: int = 0
I_RADIUS_GST_4: int = 1
I_RADIUS_GST_4prime: int = 2
I_RADIUS_GST_5: int = 3
I_RADIUS_GST_6: int = 4
I_RADIUS_DFT_5: int = 5
I_RADIUS_DFT_6: int = 6
I_RADIUS_EDT1_5: int = 7
I_RADIUS_EDT2_5: int = 8
I_RADIUS_EDT2_6: int = 9
I_RADIUS_KFT_8: int = 10


# joint indices for tendon thetas
I_THETA_GST_3: int = 0
I_THETA_GST_4: int = 1
I_THETA_GST_5: int = 2
I_THETA_ALL_6: int = 3
I_THETA_DFT_5: int = 4
I_THETA_EDT1_4: int = 5
I_THETA_EDT1_5: int = 6
I_THETA_EDT2_4: int = 7
I_THETA_EDT2_5: int = 8
I_THETA_KFT_3: int = 9
I_THETA_KFT_8: int = 10

# joint indices for q offsets
I_Q_GST_3: int = 0
I_Q_GST_4: int = 1
I_Q_GST_5: int = 2
I_Q_GST_6: int = 3
I_Q_DFT_5: int = 4
I_Q_DFT_6: int = 5

I_QHAT_EDT2_6: int = 0  # in relation to theta_6_hat


# indices for tendon tangency angles
I_TENDON_TANGENCY_ANGLE_GST_34_J4: int = 0
I_TENDON_TANGENCY_ANGLE_GST_4PRIME5_J4: int = 1
I_TENDON_TANGENCY_ANGLE_GST_4PRIME5_J5: int = 2
I_TENDON_TANGENCY_ANGLE_GST_67_J6: int = 3
I_TENDON_TANGENCY_ANGLE_DFT_56_J5: int = 4
I_TENDON_TANGENCY_ANGLE_DFT_6C_J6: int = 5
I_TENDON_TANGENCY_ANGLE_EDT1_5C_J5: int = 6
I_TENDON_TANGENCY_ANGLE_EDT2_56_J5: int = 7

# indices for tendon section lengths
I_TENDON_SECTION_LENGTH_GST_23: int = 0
I_TENDON_SECTION_LENGTH_GST_34: int = 1
I_TENDON_SECTION_LENGTH_GST_4PRIME5: int = 2
I_TENDON_SECTION_LENGTH_GST_56: int = 3
I_TENDON_SECTION_LENGTH_GST_67: int = 4
I_TENDON_SECTION_LENGTH_DFT_C5: int = 5
I_TENDON_SECTION_LENGTH_DFT_56: int = 6
I_TENDON_SECTION_LENGTH_DFT_6C: int = 7
I_TENDON_SECTION_LENGTH_EDT1_5C: int = 8
I_TENDON_SECTION_LENGTH_EDT2_56: int = 9
I_TENDON_SECTION_LENGTH_EDT2_6C: int = 10
