# sd2_epi_depthlm_e12 with the CURRENT ego status only (no ego history), for closed-loop use:
# the ego MLP sees only the current CAN bus (orientation, accel, yaw rate, velocity) + command;
# the past-pose memory and the past-position PE tokens are dropped (ego_status_len=0, no 'PE').
_base_ = ['./sd2_epi_depthlm_e12.py']

model = dict(ego_status='feature', ego_status_len=0)
