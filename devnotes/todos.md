# Todos

## Planning

- [ ] Integrate aim/weights & biases (+++)
- [ ] Automate training runs, parameter sweeps (++)
- [ ] Look into how to save a database from isaacsim (++)
- [ ] Look into how to obtain the required forces to train LNNs (++)
- [ ] Decide the IsaacNext -> Identix state mapping before recording (+++)
- [ ] Define whether the first Identix dataset uses one leg or left/right legs as separate samples (+++)
- [ ] Define the source and sign convention for Identix q = [y, theta1, theta2] (+++)
- [ ] Decide what tau means in the recorded dataset: actuator torque, total generalized force, or residual target (+++)

- [ ] Update the readme (+)
- [ ] Clean up the repo -> Make branches stale, put legacy code away (+)
- [ ] Visualize the tendons in isaacsim (+)
- [ ] Document the USD creation
- [ ] Set up a server to run experiments & test the setup code

## In progress

- [ ] Write tests to test the most important functionality
- [ ] Talk with chatGPT to improve learning / env (+++)
- [ ] Ask chatgpt about how to run better projects (+++)

## Now

- [ ] Finish setting up IsaacNext on this computer, if needed
- [ ] Work on branch feature/identix-data-recording
- [ ] Create a reusable DataRecording class under the tendon/IsaacNext code, not only inside run.py (+++)
- [ ] Base the first recording pass on scripts/tendons/run.py or a dedicated record_identix.py script (+++)
- [ ] Reuse the existing logical DOF mapping to select sagittal hip_flexion and knee_flexion only (+++)
- [ ] Exclude hip_roll and hip_yaw/hip_lateral from the first dataset (+++)
- [ ] Record Identix-compatible sim_data rows: q*, dq*, ddq*, tau* (+++)
- [ ] Record metadata sidecars: sim_dt, joint names, joint indices, signs, offsets, parameter file, controller,
  constraint mode (++)
- [ ] Extract required simulation parameters through clean interfaces instead of ad hoc script logic (++)
- [ ] Populate the Identix kinematics database first and validate it on a tiny dataset (+++)
- [ ] Populate the dynamics database with row-aligned tau component columns when force semantics are clear (+++)
- [ ] Validate units, finite values, q/dq/ddq consistency, signs, and physical meaning before scaling up (+++)
- [ ] Integrate recording with PSO only after the standalone tendon recording path is validated (++)

## Done

- [x] Create a new USD (Make a list of mods for everything first) (DOCUMENT) (++)
- [x] Build a tool to properly choose the lengths of the tendons (++)
- [x] Solve the issue with the base name (+)
- [x] Build a run script with a fake boom (+++)
- [x] Centralized config file (+++)
- [x] Solve the problems with the sources and warnings in pycharm from isaac (+++)
- [x] Updating isaac's repo (+)
- [x] Automate the linting + pycharm handling (+++)
