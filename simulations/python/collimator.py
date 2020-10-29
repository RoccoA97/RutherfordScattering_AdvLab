class Collimator():
    def __init__(self, c_x, c_y, c_z, c_w, c_h):
        self.collimator_x = c_x    # x origin    +++++
        self.collimator_y = c_y    # y origin    +   +
        self.collimator_z = c_z    # z origin    o++++
        self.collimator_w = c_w    # width
        self.collimator_h = c_h    # height
