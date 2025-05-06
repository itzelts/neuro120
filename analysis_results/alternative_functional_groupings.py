# Alternative functional groupings based on different cluster solutions

# Original groupings
original_groups = {
    'ankle':'MJA',
    'eye':'TOR',
    'finger':'DFM',
    'forearm':'PLM',
    'jaw':'OFC',
    'leftleg':'PLM',
    'lip':'OFC',
    'rightleg':'PLM',
    'toe':'DFM',
    'tongue':'OFC',
    'upperarm':'PLM',
    'wrist':'MJA',
}

# Optimal cluster solution (likely 2 clusters)
optimal_groups = {
    'ankle':'M1_PM_FOOT_JOINT',
    'eye':'PP_SM_VISUAL_CONTROL',
    'finger':'M1_PM_HAND_FINE',
    'forearm':'M1_PM_ARM_CONTROL',
    'jaw':'PP_SM_FACE_CONTROL',
    'leftleg':'PP_SM_LEG_CONTROL',
    'lip':'PP_SM_FACE_CONTROL',
    'rightleg':'PP_SM_LEG_CONTROL',
    'toe':'M1_PM_FOOT_FINE',
    'tongue':'PP_SM_ORAL_CONTROL',
    'upperarm':'M1_PM_ARM_CONTROL',
    'wrist':'M1_PM_HAND_JOINT',
}

# 4-cluster solution (maximum granularity)
granular_groups = {
    'ankle':'FOOT_JOINT',
    'eye':'VISUAL_CONTROL',
    'finger':'HAND_FINE',
    'forearm':'ARM_CONTROL',
    'jaw':'FACE_CONTROL',
    'leftleg':'LEG_CONTROL',
    'lip':'FACE_CONTROL',
    'rightleg':'LEG_CONTROL',
    'toe':'FOOT_FINE',
    'tongue':'ORAL_CONTROL',
    'upperarm':'ARM_CONTROL',
    'wrist':'HAND_JOINT',
}
