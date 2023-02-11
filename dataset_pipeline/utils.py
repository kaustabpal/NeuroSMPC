import torch
import numpy as np

def rotate(origin, point, angle=180):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return -qx+127, qy-127

def draw_circle(x, y, radius):
    th = torch.arange(0,2*np.pi,0.01)
    xunit = radius * torch.cos(th) + x
    yunit = radius * torch.sin(th) + y
    return xunit, yunit 
