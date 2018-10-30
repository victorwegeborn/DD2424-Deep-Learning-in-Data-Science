_in = open("interpolated.csv", "r");
_out_train = open("train.csv", "w")
_out_test = open("test.csv", "w")
_out_validation = open("validation.csv", "w")
line = 0
first = True
for l in _in.readlines():
    _out_file = _out_train
    if line % 10 < 7:
        _out_file = _out_train
    elif line % 10 == 8:
        _out_file = _out_validation
    else:
        _out_file = _out_test 
    #_out_file = _out_test if line % 10 == 0 else _out_train
    if first == True:
        first = False
        _out_test.write("frame_id,steering_angle,label\n")
        _out_validation.write("frame_id,steering_angle,label\n")
        _out_train.write("frame_id,steering_angle,label\n")
    else:
        cols = l.split(",")
        camera = cols[4].strip()
        if camera != "center_camera":
            continue

        filePath = cols[5].strip()
        filePath ="HMB_1/center/"+filePath.split("/")[1]
        angle = float(cols[6].strip())
        _class = 1 
        if angle < -0.02:
            _class = 0
            _out_file.write(str(filePath) + ", " + str(angle) + ", " + str(_class) + "\n")
            line +=1
        elif angle > 0.02:
            _class = 2
            _out_file.write(str(filePath) + ", " + str(angle) + ", " + str(_class) + "\n")
            line +=1
        else:
            _out_file.write(str(filePath) + ", " + str(angle) + ", " + str(_class) + "\n")
            line +=1
        #else:
        #    line +=1


