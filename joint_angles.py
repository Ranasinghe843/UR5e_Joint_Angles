import matplotlib.pyplot as plt
import numpy as np
import modern_robotics as mr
import sys
import pandas as pd
from scipy.interpolate import interp1d

def main():

    bow_data = pd.read_csv("mocap2.csv")

    # Extract the columns
    bow_x = bow_data["Bow_X"]
    bow_y = bow_data["Bow_Y"]
    bow_z = bow_data["Bow_Z"]
    rec_t = bow_data["Time"]

    # Adjust the bow data
    ad_bow_x = (bow_x - bow_x.iloc[0]) / 100
    ad_bow_y = (bow_y - bow_y.iloc[0]) / 100
    ad_bow_z = (bow_z - bow_z.iloc[0]) / 100

    # Define the time step and generate the time vector
    dt = 0.002
    t = np.arange(0, rec_t.iloc[-1], dt)

    # Interpolate the adjusted bow data using cubic splines
    interp_bow_x = interp1d(rec_t, ad_bow_x, kind="cubic", fill_value="extrapolate")
    interp_bow_y = interp1d(rec_t, ad_bow_y, kind="cubic", fill_value="extrapolate")
    interp_bow_z = interp1d(rec_t, ad_bow_z, kind="cubic", fill_value="extrapolate")

    # Generate the interpolated data
    inter_bow_x = interp_bow_x(t)
    inter_bow_y = interp_bow_y(t)
    inter_bow_z = interp_bow_z(t)

    L1 = 0.2435
    L2 = 0.2132
    W1 = 0.1311
    W2 = 0.0921
    H1 = 0.1519
    H2 = 0.0854

    M = np.array([[-1, 0, 0, L1 + L2],
                  [0, 0, 1, W1 + W2],
                  [0, 1, 0, H1 - H2],
                  [0, 0, 0, 1]])
    
    S1 = np.array([0, 0, 1, 0, 0, 0])
    S2 = np.array([0, 1, 0, -H1, 0, 0])
    S3 = np.array([0, 1, 0, -H1, 0, L1])
    S4 = np.array([0, 1, 0, -H1, 0, L1 + L2])
    S5 = np.array([0, 0, -1, -W1, L1+L2, 0])
    S6 = np.array([0, 1, 0, H2-H1, 0, L1+L2])
    S = np.array([S1, S2, S3, S4, S5, S6]).T
    
    B1 = np.linalg.inv(mr.Adjoint(M))@S1
    B2 = np.linalg.inv(mr.Adjoint(M))@S2
    B3 = np.linalg.inv(mr.Adjoint(M))@S3
    B4 = np.linalg.inv(mr.Adjoint(M))@S4
    B5 = np.linalg.inv(mr.Adjoint(M))@S5
    B6 = np.linalg.inv(mr.Adjoint(M))@S6
    B = np.array([B1, B2, B3, B4, B5, B6]).T

    theta0 = np.array([-1.6800, -1.4018, -1.8127, -2.9937, -0.8857, -0.0696])
    
    # perform forward kinematics
    T0_space = mr.FKinSpace(M, S, theta0)
    print(f'T0_space: {T0_space}')
    T0_body = mr.FKinBody(M, B, theta0)
    print(f'T0_body: {T0_body}')
    T0_diff = T0_space - T0_body
    print(f'T0_diff: {T0_diff}')
    T0 = T0_body

    # calculate Tsd for each time step
    Tsd = np.zeros((4, 4, len(t)))
    for i in range(len(t)):
        Tsd[:, :, i] = T0 @ np.array(mr.RpToTrans(np.eye(3), np.array([inter_bow_x[i], inter_bow_y[i], inter_bow_z[i]])))
        
    # plot p(t) vs t in the {s} frame
    xs = Tsd[0, 3, :]
    ys = Tsd[1, 3, :]
    zs = Tsd[2, 3, :]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, 'b-',label='p(t)')
    ax.plot(xs[0], ys[0], zs[0], 'go',label='start')
    ax.plot(xs[-1], ys[-1], zs[-1], 'rx',label='end')
    ax.set_title('Trajectory in s frame')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend()
    plt.show()

    # when i=0
    thetaAll = np.zeros((6, len(t)))

    initialguess = theta0
    eomg = 1e-6
    ev = 1e-6

    thetaSol, success = mr.IKinBody(B, M, Tsd[:,:,0], initialguess, eomg, ev)
    if not success:
        raise Exception(f'Failed to find a solution at index {0}')
    thetaAll[:, 0] = thetaSol

    # when i=1...,N-1
    for i in range(1, len(t)):
        initialguess = thetaAll[:, i-1]

        thetaSol, success = mr.IKinBody(B, M, Tsd[:,:,i], initialguess, eomg, ev)
        if not success:
            raise Exception(f'Failed to find a solution at index {i}')
        thetaAll[:, i] = thetaSol

    # verify that the joint angles don't change much
    dj = np.diff(thetaAll, axis=1)
    plt.plot(t[1:], dj[0], 'b-',label='joint 1')
    plt.plot(t[1:], dj[1], 'g-',label='joint 2')
    plt.plot(t[1:], dj[2], 'r-',label='joint 3')
    plt.plot(t[1:], dj[3], 'c-',label='joint 4')
    plt.plot(t[1:], dj[4], 'm-',label='joint 5')
    plt.plot(t[1:], dj[5], 'y-',label='joint 6')
    plt.xlabel('t (seconds)')
    plt.ylabel('first order difference')
    plt.title('Joint angles first order difference')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # verify that the joint angles will trace out our trajectory
    actual_Tsd = np.zeros((4, 4, len(t)))
    for i in range(len(t)):
        actual_Tsd[:,:,i] = mr.FKinBody(M, B, thetaAll[:, i])
    
    xs = actual_Tsd[0, 3, :]
    ys = actual_Tsd[1, 3, :]
    zs = actual_Tsd[2, 3, :]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, 'b-',label='p(t)')
    ax.plot(xs[0], ys[0], zs[0], 'go',label='start')
    ax.plot(xs[-1], ys[-1], zs[-1], 'rx',label='end')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Verified Trajectory in s frame')
    ax.legend()
    plt.show()

    # save to csv file (you can modify the led column to control the led)
    # led = 1 means the led is on, led = 0 means the led is off
    data = np.column_stack((t, thetaAll.T))
    np.savetxt('file.csv', data, delimiter=',')


if __name__ == "__main__":
    main()