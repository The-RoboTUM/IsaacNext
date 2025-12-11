import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

with open("outputs/joint_pos_gst.json", "r") as f:
    gst = json.load(f)


gst_j1 = [x[0] for x in gst][:210]
gst_j2 = [x[1] for x in gst][:210]
gst_j3 = [x[2] for x in gst][:210]
gst_j4 = [x[3] for x in gst][:210]
gst_j5 = [x[4] for x in gst][:210]
gst_j6 = [x[5] for x in gst][:210]


fig = plt.figure(figsize=(5, 10))
axs = fig.subplots(1, 1)


axs.plot(list(range(len(gst_j1))), gst_j1, color="blue")
axs.plot(list(range(len(gst_j2))), gst_j2, color="green")
axs.plot(list(range(len(gst_j3))), gst_j3, color="orange")
axs.plot(list(range(len(gst_j4))), gst_j4, color="yellow")
axs.plot(list(range(len(gst_j5))), gst_j5, color="purple")
axs.plot(list(range(len(gst_j6))), gst_j6, color="pink")


plt.show()
