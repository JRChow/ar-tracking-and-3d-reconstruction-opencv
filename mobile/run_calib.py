import CalibrationHelpers as calib
intrinsics, distortion, roi, new_intrinsics = \
    calib.CalibrateCamera('calibration_data', True)
print("intrinsics = \n", intrinsics)
print("distortion = \n", distortion)
print("roi = \n", roi)
print("new_intrinsics = \n", new_intrinsics)
calib.SaveCalibrationData('calibration_data', intrinsics, distortion, new_intrinsics, roi)
