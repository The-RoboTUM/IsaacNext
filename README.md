# IsaacNext

Short description of what this project does and who it’s for.

---
## How to obtain Forrest's URDF
You can find the Forrest's URDF file from this link:\
https://github.com/The-RoboTUM/urdfheim/tree/main/complex/Forrest_URDF_description/urdf 

You `git clone` the whole repo and go the above folder and find the urdf file.

In Isaacsim, we need to USD file, so we need to convert URDF file to USD file.

Nest, let's go through the operation steps in detail:

## How to prepare Forrest's USD


- Open Isaacsim
- Import the URDF file (change these settings)
![See the screenshot](images/urdf_import_setting.png)
  - Choose `Moveable Base`
  - Create `Collisions From Visuals`
  - Choose `Convex Decomposition`
- after you finish doing above steps, Isaacsim will generate a folder: `Forrest_URDF`, at the same place of the urdf file.
- Move only the generated folder `Forrest_URDF` over to the symlinks (In the future a script will do this)
- In Isaacsim, open `/IsaacNext/symlinks/Forrest_URDF/Forrest_URDF.usd`
- After you open the USD file, then on the right-hand Stage panel in IsaacSim, find the joints folder:
-![](images/joints%20folder%20in%20the%20IsaacSim.png)
  - For all set stiffness to 100 and damping to 1
  - For anchor joints disable them from articulation
  - For anchor joints set max force to 1000, stiffness to 1000 and damping to 10
  - For passive pantograph joints (r3b, r4f, r4b, l3b, l4f, l4b) set the stiffness to 0
  - Set the offsets of all the joints to the list bellow (the **Joints limits** part)
- Use stage search for 'collisions' and for all disable instantiable flag
- Use stage search for 'mesh', for the active items, look all the bodies of the pantograph and the inner gears of the hip and disable collisions for them
- Testing set a ground and simulate, check collisions
- Go to joints and verify that all joints can be actuated

### Joints limits
- `l0_acetabulofemoral_roll` = [-180, 180]
- `l1_acetabulofemoral_lateral` = [-10, 10]
- `l2_pseudo_acetabulofemoral_flexion` = [-120, 120]
- `l3b_femorotibial_back` = [-180, 180]
- `l4b_intertarsal_back` = [-180, 35]
- `l3f_femorotibial_front` = [-76, 18]
- `l4f_intertarsal_front` = [-180, 180]
- `Virtual_S23_Assyv18_mirror_1_anchor` = [-0.0573, 0.0573]
- `l5_metatarsophalangeal` = [-20, 40]
- `l6_interphalangeal` = [-100, 90]
- `l4p_intertarsal_pulley` = [-180, 180]
- `l2p_acetabulofemoral_pulley` = [-180, 180]
- `l2b_acetabulofemoral_flexion` = [-180, 180]
- `l2f_acetabulofemoral_flexion` = [-180, 180]
- `r0_acetabulofemoral_roll` = [-180, 180]
- `r1_acetabulofemoral_lateral` = [-10, 10]
- `r2_pseudo_acetabulofemoral_flexion` = [-120, 120]
- `r3b_femorotibial_back` = [-180, 180]
- `r4b_intertarsal_back` = [-180, 180]
- `r3f_femorotibial_front` = [-76, 18]
- `r4f_intertarsal_front` = [-180, 180]
- `r4p_intertarsal_pulley` = [-180, 180]
- `Virtual_S23_Assyv18_1_anchor` = [-0.0573, 0.0573]
- `r5_metatarsophalangeal` = [-20, 40]
- `r6_interphalangeal` = [-100, 90]
- `r2p_acetabulofemoral_pulley` = [-180, 180]
- `r2b_acetabulofemoral_flexion` = [-180, 180]
- `r2f_acetabulofemoral_flexion` = [-180, 180]

---

## 📦 Installation

```bash
# Clone the repository
git clone git@github.com:your-org/your-repo.git

# Enter the project directory
cd your-repo

# Install dependencies (example for Python)
pip install -r requirements.txt
