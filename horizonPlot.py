import numpy as np
import matplotlib.pyplot as plt

#Individual samples
MAPEL = np.array([5.89, 9.26, 11.44, 13.14, 14.57, 16.99, 16.37, 18.51, 19.19, 18.15, 18.84, 19.53, 20.03, 20.52, 25.7])
VarL = np.array([[5.89, 6.41, 6.42, 6.89, 7.42, 10.97, 7.93, 10.92, 11.28, 7.78, 8.85, 9.36, 8.31, 8.25, 25.72], [5.89, 11.73, 14.86, 17.07, 18.8, 21.12, 20.81, 22.67, 23.43, 22.79, 23.15, 23.57, 23.96, 24.03, 25.64]])
VarL = np.abs(VarL - MAPEL)
NL = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90]
MAPEG = np.array([5.8, 9.1, 11.59, 13.78, 14.57, 15.81, 16.33, 17.23, 17.93, 18.27, 19.38, 19.56, 20.09, 20.83, 21.77, 21.75, 21.75, 21.97, 21.93, 22.32, 22.39, 22.68, 22.60, 23, 25.69])
VarG = np.array([[5.8, 6.29, 6.62, 7.7, 7.70, 8.03, 7.46, 7.42, 8.15, 8.51, 9.75, 7.51, 7.50, 10.36, 8.75, 8.9, 10.96, 7.98, 8.00, 8.78, 9.04, 8.42, 8.81, 8.76, 25.63], [5.8, 11.59, 15, 17.64, 18.75, 20.22, 20.96, 21.74, 22.41, 22.68, 23.44, 23.6, 23.68, 24.81, 24.02, 24.63, 24.01, 24.58, 24.69, 24.63, 24.86, 24.85, 24.85, 25.17, 25.65]])
NG = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 190, 200]
VarG = np.abs(VarG - MAPEG)
plt.figure()
plt.plot(NL, MAPEL, "b.", markersize = 10, linestyle ="-", label = "LSTM")
plt.errorbar(NL, MAPEL, yerr = VarL, color = "b", capsize=4, elinewidth = 1, capthick=1)
plt.plot(NG, MAPEG, "r.", markersize = 10, linestyle ="-", label = "GRU")
plt.errorbar(NG, MAPEG, yerr = VarG, color = "r", capsize=4, elinewidth = 1, capthick=1)
plt.grid()
plt.legend()
plt.ylabel("MAPE (%)")
plt.xlabel("Future samples predicted")

#Smoothed samples
MAPEL = np.array([0.87, 5.29, 9.81, 12.56, 14.1, 15.38, 16.5, 17.35, 17.78, 18.31, 18.71, 18.78, 19.28, 19.95, 19.99, 20.30, 22.69])
VarL = np.array([[0.87, 1.02, 1.64, 1.56, 1.46, 1.84, 2.08, 1.6, 1.91, 1.79, 2.64, 1.84, 1.77, 4.01, 2.85, 2.91, 22.65], [0.87, 9.79, 15.74, 18.77, 19.98, 20.92, 21.28, 21.84, 21.98, 22.08, 22.1, 22.25, 22.21, 22.41, 22.57, 22.53, 22.66]])
VarL = np.abs(VarL - MAPEL)
NL = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200]
MAPEG = np.array([0.87, 5.05, 9.69, 12.49, 14.52, 15.72, 16.82, 17.54, 17.85, 18.31, 18.75, 19.23, 19.08, 19.75, 20.13, 20.11, 20.48, 20.59, 22.36])
VarG = np.array([[0.87, 0.95, 1.16, 1.55, 2.14, 2.17, 1.99, 2.21, 2.59, 2.56, 2.22, 2.3, 2.46, 3.24, 4.27, 3.06, 3.61, 3.02, 22.25], [0.87, 9.56, 15.75, 18.57, 20.16, 21.34, 21.90, 21.91, 21.73, 21.94, 22.03, 22.04, 21.98, 22.25, 22.14, 22.62, 22.55, 22.8, 22.31]])
NG = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240]
VarG = np.abs(VarG - MAPEG)
plt.figure()
plt.plot(NL, MAPEL, "b.", markersize = 10, linestyle ="-", label = "LSTM")
plt.errorbar(NL, MAPEL, yerr = VarL, color = "b", capsize=4, elinewidth = 1, capthick=1)
plt.plot(NG, MAPEG, "r.", markersize = 10, linestyle ="-", label = "GRU")
plt.errorbar(NG, MAPEG, yerr = VarG, color = "r", capsize=4, elinewidth = 1, capthick=1)
plt.grid()
plt.legend()
plt.ylabel("MAPE (%)")
plt.xlabel("Future samples predicted")

#Smoothed and subsampled samples
MAPEL = np.array([5.54, 12.59, 15.96, 18.74, 19.88, 20.59, 22.92])
VarL = np.array([[5.54, 5.99, 5.87, 6.06, 6.21, 6.67, 22.91], [5.54, 17.06, 20.33, 21.36, 22.08, 22.67, 22.91]])
VarL = np.abs(VarL - MAPEL)
NL = [1, 5, 10, 20, 30, 40, 50]
MAPEG = np.array([5.54, 12.53, 16, 18.61, 19.75, 20.51, 23])
VarG = np.array([[5.54, 5.85, 5.93, 6.19, 6.25, 7.19, 23], [5.54, 16.89, 20.46, 21.52, 22.13, 22.58, 22.99]])
NG = [1, 5, 10, 20, 30, 40, 50]
VarG = np.abs(VarG - MAPEG)
plt.figure()
plt.plot(NL, MAPEL, "b.", markersize = 10, linestyle ="-", label = "LSTM")
plt.errorbar(NL, MAPEL, yerr = VarL, color = "b", capsize=4, elinewidth = 1, capthick=1)
plt.plot(NG, MAPEG, "r.", markersize = 10, linestyle ="-", label = "GRU")
plt.errorbar(NG, MAPEG, yerr = VarG, color = "r", capsize=4, elinewidth = 1, capthick=1)
plt.grid()
plt.legend()
plt.ylabel("MAPE (%)")
plt.xlabel("Future samples predicted")
plt.show()

