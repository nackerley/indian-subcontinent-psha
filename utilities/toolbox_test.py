# -*- coding: utf-8 -*-
#
# Indian Subcontinent PSHA
# Copyright (C) 2016-2018 Nick Ackerley
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
The beginnings of regression testing.
'''
import toolbox as tb
import numpy as np

assert tb.stdval(np.pi) == 3.16

test = [736, 0.548, 91.1]
result = tb.stdval(test, 192)
assert np.allclose(result,[732, 0.549, 90.9])

test = np.array([[  57],
       [-428],
       [ 409]])
result = tb.stdval(test, 12)
assert all((result == np.array([[56], [np.NaN], [ 390]])) \
    | (np.isnan(result) & np.array([[False], [True], [False]], dtype=bool)))

test = np.array([[  6.863e-02,   3.214e-02,   2.997e+00],
       [  9.320e-02,   6.416e+01,   3.428e+02]])
result = tb.stdval(test, 24, 0.5)
assert np.allclose(result,
    np.array([[  7.5e-02,   3.3e-02,   3.0e+00],
       [  1.0e-01,   6.8e+01,   3.6e+02]]))

test = np.array([[[  3.62e+02,   1.47e-02,   3.15e+01,   1.71e-03],
        [  6.94e-01,   1.61e-02,   3.04e-02,   4.79e+01],
        [  1.14e+02,   3.17e-03,   3.89e+01,   1.51e+01]],
       [[  4.71e+00,   4.34e+02,   2.31e+00,   7.67e+00],
        [  1.34e-03,   8.16e+00,   8.14e+00,   5.57e-02],
        [  5.34e-01,   2.46e+02,   4.18e-01,   1.32e+00]]])
result = tb.stdval(test, preferred=[1, 3, 10])
assert np.allclose(result,
    np.array([[[   3e+02,    1e-02,    3e+01,    1e-03],
        [   1e+00,    1e-02,    3e-02,    3e+01],
        [   1e+02,    3e-03,    3e+01,    1e+01]],
       [[   3e+00,    3e+02,    3e+00,    1e+01],
        [   1e-03,    1e+01,    1e+01,    1e-01],
        [   3e-01,    3e+02,    3e-01,    1e+00]]]))