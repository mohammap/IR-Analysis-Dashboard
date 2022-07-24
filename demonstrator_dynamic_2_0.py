### updates
### - image generated from the full picture


import shutil
import sys
import time
import logging
from shutil import copyfile

import scipy
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
from matplotlib import pyplot as plt
import math
import os
import sys
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
import time
import os.path
from matplotlib.lines import Line2D

### Deployment
if os.path.isfile('masterfile_instant.csv'):
    print ("File exists")
else:
    #if headers are neede in the masterfile
    csv_header = ["Filename", "Mean value", 'Standard deviation', '3D homogeneity', 'Homogeneity percentage',
                  'Skewness', 'Kurtosis', 'Mean homogeneity error?', 'Homogeneity line',
                  'Size of cropped tool in x [mm]',
                  'Size of cropped tool in y [mm]', 'File name_part', 'Mean Values_part', 'Standard deviation_part',
                  '3D homogeneity_part', 'Homogeneity percentage part', 'Skewness_part', 'Kurtosis_part',
                  'Mean homog. error_part', 'Homogeneity line_part', 'Size of cropped part in x [mm]',
                  'Size of cropped part in y [mm]', 'Warning system']
    e = pd.DataFrame(columns=csv_header)
    e.to_csv('masterfile_instant.csv', mode='a', index=False)  # if needed: here write csv header


if os.path.isfile('masterfile_instant_demo.csv'):
    print ("File exists")
else:
    #if headers are neede in the masterfile
    csv_header = ["Filename", "Mean value", 'Standard deviation', '3D homogeneity', 'Homogeneity percentage',
                  'Skewness', 'Kurtosis', 'Mean homogeneity error?', 'Homogeneity line',
                  'Size of cropped tool in x [mm]',
                  'Size of cropped tool in y [mm]']
    e = pd.DataFrame(columns=csv_header)
    e.to_csv('masterfile_instant_demo.csv', mode='a', index=False)  # if needed: here write csv header

if os.path.isfile('masterfile_instant_part.csv'):
        print("File exists")
else:
        # if headers are neede in the masterfile
        csv_header = ["Filename", "Mean value", 'Standard deviation', '3D homogeneity', 'Homogeneity percentage',
                      'Skewness', 'Kurtosis', 'Mean homogeneity error?', 'Homogeneity line',
                      'Size of cropped tool in x [mm]',
                      'Size of cropped tool in y [mm]', 'Warning system']
        e = pd.DataFrame(columns=csv_header)
        e.to_csv('masterfile_instant_part.csv', mode='a', index=False)  # if needed: here write csv header

if not os.path.isdir("plots/"):
    os.makedirs("plots/")
else:
    pass

if not os.path.isdir("plots/dashboard/"):
    os.makedirs("plots/dashboard/")
else:
    pass


if not os.path.isdir("digested/"):
    os.makedirs("digusted/")
else:
    pass

if not os.path.isdir("plots/thermal_image/"):
    os.makedirs("plots/thermal_image/")
else:
    pass

if not os.path.isdir("plots/dashboard/thermal_image/"):
    os.make.dirs("plots/dashboard/thermal_image/")
else:
    pass

if not os.path.isdir("plots/homline/"):
    os.makedirs("plots/homline/")
else:
    pass

if not os.path.isdir("plots/homline_part/"):
    os.makedirs("plots/homline_part/")
else:
    pass

if not os.path.isdir("plots/full_tool/"):
    os.makedirs("plots/full_tool/")
else:
    pass

if not os.path.isdir("plots/cropped_tool/"):
    os.makedirs("plots/cropped_tool/")
else:
    pass

if not os.path.isdir("plots/cropped_tool_mm/"):
    os.makedirs("plots/cropped_tool_mm/")
else:
    pass

if not os.path.isdir("plots/cropped_tool_with_temp_mm/"):
    os.makedirs("plots/cropped_tool_with_temp_mm/")
else:
    pass

if not os.path.isdir("plots/part/"):
    os.makedirs("plots/part/")
else:
    pass

if not os.path.isdir("plots/threshold/"):
    os.makedirs("plots/threshold/")
else:
    pass

if not os.path.isdir("plots/histograms/"):
    os.makedirs("plots/histograms/")
else:
    pass

if not os.path.isdir("plots/histograms_with_distributions/"):
    os.makedirs("plots/histograms_with_distributions/")
else:
    pass

if not os.path.isdir("plots/full_tool_by_area/"):
    os.makedirs("plots/full_tool_by_area/")
else:
    pass

if not os.path.isdir("plots/dashboard/full_tool_by_area/"):
    os.makedirs("plots/dashboard/full_tool_by_area/")
else:
    pass

if not os.path.isdir("plots/part_dynamic/"):
    os.makedirs("plots/part_dynamic/")
else:
    pass

if not os.path.isdir("plots/dashboard/part_dynamic/"):
    os.makedirs("plots/dashboard/part_dynamic/")
else:
    pass

class Event(LoggingEventHandler):
    def on_created(self, event):

        import math
        import os
        from shutil import copyfile
        import sys
        from matplotlib import pyplot as plt
        from scipy.stats import skew, kurtosis
        import numpy as np
        import pandas as pd
        import time
        #import cv2
        import shutil
        # Conversion script
        start_time2 = time.time()
        meanarray_tool = []  # Creates a new array for mean values
        filenamearray_tool = []  # Creates a new array for filenames
        homoarray_tool = []
        st_dev_tool = []
        skewarray_tool = []
        kurtarray_tool = []
        meanerror_array_tool = []
        homogeneity_line_array_tool = []
        w_tool_array = []
        h_tool_array = []
        homogeneity_percentage_tool_array = []

        meanarray_part = []  # Creates a new array for mean values
        filenamearray_part = []  # Creates a new array for filenames
        homoarray_part = []
        st_dev_part = []
        skewarray_part = []
        kurtarray_part = []
        meanerror_array_part = []
        homogeneity_line_array_part = []
        w_part_array = []
        h_part_array = []
        homogeneity_percentage_part_array = []
        warning_zero_part = []

        ### Inputs ###
        real_tool_size_x = 880
        real_tool_size_y = 380
        real_tool_crop_size = 20  # How many mm should be cropped from each side
        real_part_size_x = 230
        real_part_size_y = 150
        threshold_limit = 10  # this value is substracted from the average T of the part which is used for thresholding
        # Drawing constants
        cross_size = 10  # was 20 for smaller pictures - if it should be back to constant values
        ThreeD_graph = 0
        old_crop = 1
        below_temperature_delete = 100 # temperatures below this limit is not considered
        w_crop_control = 0.5  # controls how many percentage of "0"s should be considered in the cropping process
        h_crop_control = 0.6  # e.g: 400 rows, morethan 60% contains 0 -> delete
        set_temperature = 200
        percentage_for_ok_not_ok = 5



        img_number = 1  # counter for stacking meanarray,homoarray, etc
        for root, dirs, files in os.walk("excel/"):
            for name in files:
                import math
                import os
                from matplotlib import pyplot as plt
                from scipy.stats import skew, kurtosis
                import numpy as np
                import pandas as pd
                import time
                from mpl_toolkits.mplot3d import Axes3D
                start_time = time.time()
                print(os.path.join(root, name))  # will print path of files
                path = os.path.join(root, name)
                if path.endswith((".xlsx", ".xls")):
                    time.sleep(1)
                    excel = pd.read_excel(path,
                                          header=None,
                                          index_col=False)
                    thermo_array = excel.to_numpy(dtype=np.dtype,
                                                  na_value=0)
                else:
                    time.sleep(1)
                    excel = pd.read_csv(path,
                                        header=None,
                                        index_col=False)
                    thermo_array = excel.to_numpy(dtype=np.dtype,
                                                  na_value=0)
                copyfile("excel/" + name,
                         "digested/excel/" + name)
                # converts excel sheet to numpy
                # End of operation

                filename = os.path.splitext(name)[0]  # split the file name from the original extension
                print(filename)
                # if thermo_array[0, 0] == "Column1": #next one is more elegant
                if isinstance(thermo_array[0, 0], str):
                    thermo_array = np.delete(thermo_array, 0, 0)


                h, w = thermo_array.shape  # get shape of thermo pic
                thermo_array = thermo_array.astype(float)
                zeros_column = np.count_nonzero(thermo_array < below_temperature_delete, axis=0)  # collect number of 0 in a column
                zeros_column_max = np.amax(zeros_column)
                zeros_column_max = h * h_crop_control  # controls the cropping process

                zeros_row = np.count_nonzero(thermo_array < below_temperature_delete, axis=1)  # collect number of 0 in a row
                zeros_row_max = np.amax(zeros_row)
                zeros_row_max = w * w_crop_control  # controls the cropping process

                w_zero = zeros_column.shape  # how many zero columns are there
                w_zero_int = int(w_zero[0])
                index_column = []
                h_zero = zeros_row.shape
                h_zero_int = int(h_zero[0])
                index_row = []
                index_column.append(0)
                for i in range(w_zero_int):
                    if zeros_column[i] > zeros_column_max:
                        index_column.append(i)  # collects the index of columns which needs to be deleted
                index_row.append(
                    0)  # TODO important: this has been used for parts where the very top of the rows also contain important termal data. Here, a 0 is added to the row indexes because then the cropping will process will start from 0. If this doesnt exist, then it wont be able to define a range
                for i in range(h_zero_int):
                    if zeros_row[i] > zeros_row_max:
                        index_row.append(i)
                index_row.insert(len(index_row), h_zero_int - 1)
                index_column.insert(len(index_column), w_zero_int - 1)

                index_of_part_row = []
                index_of_part_column = []
                length_of_index_column = len(index_column)
                length_of_index_row = len(index_row)
                for i in range(length_of_index_row):  # finds the first and last row index of the part
                    if index_row[i] - index_row[i - 1] > 20:
                        first_part_row_index = index_row[i - 1] + 2
                        index_of_part_row.append(first_part_row_index)
                        index_of_part_row.append(index_row[i])
                for i in range(length_of_index_column):
                    if index_column[i] - index_column[i - 1] > 20:
                        first_part_column_index = index_column[i - 1] + 2
                        index_of_part_column.append(first_part_column_index)
                        index_of_part_column.append(index_column[i])
                # First the script finds how many cells doesnt satisfy the criteria of higher than 100°C. Then, if 60%+ of these cells are found in a row or column, they get noted. We have a list which doesnt satisy our criteria. Then, a loop determines if there is a large gap in the indexes of the row and columns. If this is more than 10 (so that our part is between eg 35 and 300) then this is collected in a list. For the first index for both rows and columbs, two has to be added because python counts from 0, and in order to be inside the part array, we have to again add 1 more index. For the last one, they cancel out: +1 -1
                length_check_of_index_of_part_column = len(index_of_part_column)
                length_check_of_index_of_part_row = len(index_of_part_column)
                if length_check_of_index_of_part_column == 0:  # TODO debugging. if the cropping is unsuccessful, the full picture will be ""analyzed"". This is to prevent break of the code
                    index_of_part_column.append(1)
                    index_of_part_column.append(w_zero_int)
                if length_check_of_index_of_part_row == 0:
                    index_of_part_row.append(1)
                    index_of_part_row.append(h_zero_int)
                part_top_left_corner_x = index_of_part_column[0] - 1
                part_top_left_corner_y = index_of_part_row[0] - 1
                part_bottom_left_corner_x = index_of_part_column[0] - 1
                part_bottom_left_corner_y = index_of_part_row[1]
                part_top_right_corner_x = index_of_part_column[1]
                part_top_right_corner_y = index_of_part_row[0] - 1
                part_bottom_right_corner_x = index_of_part_column[1]
                part_bottom_right_corner_y = index_of_part_row[1]

                # full part area call out!!!!!
                thermo_array_crop = thermo_array[part_top_left_corner_y:part_bottom_right_corner_y,
                                    part_top_left_corner_x:part_top_right_corner_x]
                thermo_array_crop = thermo_array_crop.astype(float)

                thermo_array_crop_full_tool = np.asarray(thermo_array_crop)
                thermo_array_crop_full_tool = thermo_array_crop_full_tool.astype(int)  # full tool call out
                thermo_array_crop_np = np.asarray(thermo_array_crop)
                thermo_array_crop_np = thermo_array_crop_np.astype(int)
                max_T_full_tool = np.amax(thermo_array_crop_full_tool)
                min_T_full_tool = np.amin(thermo_array_crop_full_tool)
                max_T = np.amax(thermo_array_crop_np)
                min_T = np.amin(thermo_array_crop_np)
                h_tool, w_tool = thermo_array_crop_np.shape
                # Conversion of pixels to mm
                mm_in_x = w_tool / real_tool_size_x  # 880 = tool width
                mm_in_y = h_tool / real_tool_size_y  # 380 = tool height
                size_of_crop_x = math.floor(mm_in_x * real_tool_crop_size)
                size_of_crop_y = math.floor(mm_in_y * real_tool_crop_size)
                thermo_array_crop_np = thermo_array_crop_np[size_of_crop_y:, :]  # first n rows deleted
                thermo_array_crop_np = thermo_array_crop_np[:-size_of_crop_y, :]  # last n rows deleted
                thermo_array_crop_np = thermo_array_crop_np[:, size_of_crop_x:]
                thermo_tool_crop_np = thermo_array_crop_np[:, :-size_of_crop_x]  # cropped tool call-out!!
                h_tool_crop, w_tool_crop = thermo_tool_crop_np.shape
                cropped_tool_in_x_mm = w_tool_crop / mm_in_x
                cropped_tool_in_x_mm = round(cropped_tool_in_x_mm, 3)
                cropped_tool_in_y_mm = h_tool_crop / mm_in_y
                cropped_tool_in_y_mm = round(cropped_tool_in_y_mm, 3)

                print('size of cropped tool =', cropped_tool_in_x_mm, '*', cropped_tool_in_y_mm)

                # Focus on part. Def crop_center: https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
                size_of_part_x = math.floor(mm_in_x * real_part_size_x)
                size_of_part_y = math.floor(mm_in_y * real_part_size_y)

                # if cropping manual, for reference
                # startx = w_tool//2-(size_of_part_x//2)
                # starty = h_tool//2-(size_of_part_y//2)
                # thermo_part = thermo_array_crop_np[(starty):starty+size_of_part_y,startx:startx+size_of_part_x]
                def crop_center(img, cropx, cropy):  # cropping from the center
                    y, x = img.shape
                    startx = x // 2 - (cropx // 2)
                    starty = y // 2 - (cropy // 2)
                    return img[starty:starty + cropy, startx:startx + cropx]

                    # The script below changes the axis ticks, so that they are centered, and the distance is written on them. There might some rounding error, but it is good enough

                def x_axis(w_axis):
                    x_unmodified_ticks_tool = list(range(0, w_axis))
                    x_unmodified_ticks_tool = np.array(x_unmodified_ticks_tool)
                    x_modified_ticks_tool = [item / mm_in_x for item in x_unmodified_ticks_tool]
                    x_modified_ticks_tool = [round(item, 1) for item in x_modified_ticks_tool]

                    middle0 = x_modified_ticks_tool[0]
                    middle1 = x_modified_ticks_tool[math.floor((len(x_modified_ticks_tool)) / 2)]
                    middle2 = x_modified_ticks_tool[math.floor((len(x_modified_ticks_tool) / 4))]
                    middle3 = x_modified_ticks_tool[math.floor((len(x_modified_ticks_tool) / 8))]
                    middle4 = x_modified_ticks_tool[
                        math.floor((len(x_modified_ticks_tool) / 4) + (len(x_modified_ticks_tool) / 8))]
                    middle5 = x_modified_ticks_tool[
                        math.floor((len(x_modified_ticks_tool) / 2) + (len(x_modified_ticks_tool) / 8))]
                    middle6 = x_modified_ticks_tool[
                        math.floor((len(x_modified_ticks_tool) / 2) + (len(x_modified_ticks_tool) / 4))]
                    middle7 = x_modified_ticks_tool[math.floor(
                        (len(x_modified_ticks_tool) / 2) + (len(x_modified_ticks_tool) / 4) + (
                                len(x_modified_ticks_tool) / 8))]
                    middle8 = x_modified_ticks_tool[-1] + (1 / mm_in_x)
                    middle8 = round(middle8, 1)
                    middle_list = []
                    middle_list.extend(
                        [middle0, middle3, middle2, middle4, middle1, middle5, middle6, middle7, middle8])

                    middleb0 = x_unmodified_ticks_tool[0]
                    middleb1 = x_unmodified_ticks_tool[math.floor((len(x_unmodified_ticks_tool)) / 2)]
                    middleb2 = x_unmodified_ticks_tool[math.floor((len(x_unmodified_ticks_tool) / 4))]
                    middleb3 = x_unmodified_ticks_tool[math.floor((len(x_unmodified_ticks_tool) / 8))]
                    middleb4 = x_unmodified_ticks_tool[
                        math.floor((len(x_unmodified_ticks_tool) / 4) + (len(x_unmodified_ticks_tool) / 8))]
                    middleb5 = x_unmodified_ticks_tool[
                        math.floor((len(x_unmodified_ticks_tool) / 2) + (len(x_unmodified_ticks_tool) / 8))]
                    middleb6 = x_unmodified_ticks_tool[
                        math.floor((len(x_unmodified_ticks_tool) / 2) + (len(x_unmodified_ticks_tool) / 4))]
                    middleb7 = x_unmodified_ticks_tool[math.floor(
                        (len(x_unmodified_ticks_tool) / 2) + (len(x_unmodified_ticks_tool) / 4) + (
                                len(x_unmodified_ticks_tool) / 8))]
                    middleb8 = x_unmodified_ticks_tool[-1] + 1
                    middle_list_b = []
                    middle_list_b.extend(
                        [middleb0, middleb3, middleb2, middleb4, middleb1, middleb5, middleb6, middleb7, middleb8])
                    return plt.xticks((middle_list_b), (middle_list))
                    middle_list.clear()
                    middle_list_b.clear()

                def y_axis(h_axis):
                    y_unmodified_ticks_tool = list(range(0, h_axis))
                    y_unmodified_ticks_tool = np.array(y_unmodified_ticks_tool)
                    y_modified_ticks_tool = [item / mm_in_y for item in y_unmodified_ticks_tool]
                    y_modified_ticks_tool = [round(item, 1) for item in y_modified_ticks_tool]

                    middle0 = y_modified_ticks_tool[0]
                    middle1 = y_modified_ticks_tool[math.floor((len(y_modified_ticks_tool)) / 2)]
                    middle2 = y_modified_ticks_tool[math.floor((len(y_modified_ticks_tool) / 4))]
                    middle3 = y_modified_ticks_tool[math.floor((len(y_modified_ticks_tool) / 8))]
                    middle4 = y_modified_ticks_tool[
                        math.floor((len(y_modified_ticks_tool) / 4) + (len(y_modified_ticks_tool) / 8))]
                    middle5 = y_modified_ticks_tool[
                        math.floor((len(y_modified_ticks_tool) / 2) + (len(y_modified_ticks_tool) / 8))]
                    middle6 = y_modified_ticks_tool[
                        math.floor((len(y_modified_ticks_tool) / 2) + (len(y_modified_ticks_tool) / 4))]
                    middle7 = y_modified_ticks_tool[math.floor(
                        (len(y_modified_ticks_tool) / 2) + (len(y_modified_ticks_tool) / 4) + (
                                len(y_modified_ticks_tool) / 8))]
                    middle8 = y_modified_ticks_tool[-1] + (1 / mm_in_y)
                    middle8 = round(middle8, 1)
                    middle_list = []
                    # middle_list.extend([middle0, middle2, middle1, middle6, middle8])
                    middle_list.extend([middle8, middle6, middle1, middle2, middle0])

                    middleb0 = y_unmodified_ticks_tool[0]
                    middleb1 = y_unmodified_ticks_tool[math.floor((len(y_unmodified_ticks_tool)) / 2)]
                    middleb2 = y_unmodified_ticks_tool[math.floor((len(y_unmodified_ticks_tool) / 4))]
                    middleb3 = y_unmodified_ticks_tool[math.floor((len(y_unmodified_ticks_tool) / 8))]
                    middleb4 = y_unmodified_ticks_tool[
                        math.floor((len(y_unmodified_ticks_tool) / 4) + (len(y_unmodified_ticks_tool) / 8))]
                    middleb5 = y_unmodified_ticks_tool[
                        math.floor((len(y_unmodified_ticks_tool) / 2) + (len(y_unmodified_ticks_tool) / 8))]
                    middleb6 = y_unmodified_ticks_tool[
                        math.floor((len(y_unmodified_ticks_tool) / 2) + (len(y_unmodified_ticks_tool) / 4))]
                    middleb7 = y_unmodified_ticks_tool[math.floor(
                        (len(y_unmodified_ticks_tool) / 2) + (len(y_unmodified_ticks_tool) / 4) + (
                                len(y_unmodified_ticks_tool) / 8))]
                    middleb8 = y_unmodified_ticks_tool[-1] + 1
                    middle_list_b = []
                    middle_list_b.extend([middleb0, middleb2, middleb1, middleb6, middleb8])

                    plt.gca().invert_yaxis()
                    return plt.yticks((middle_list_b), (middle_list))
                    middle_list.clear()
                    middle_list_b.clear()
                ### drawing of full picture
                plt.pcolormesh(thermo_array, cmap="viridis", vmin=0, vmax=max_T)
                plt.axis('scaled')
                plt.title("Thermal image of "+str(filename))

                x_axis(w)
                y_axis(h)

                plt.grid('major')
                plt.colorbar()
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(12, 8)
                plt.savefig('plots/thermal_image/' + str(filename) + "_thermal_image" + ".png", bbox_inches='tight',
                            dpi=200)
                plt.savefig('plots/dashboard/thermal_image/' + "thermal_image" + ".png", bbox_inches='tight',
                            dpi=200)
                # plt.show()
                plt.clf()



                thermo_array_crop_part = crop_center(thermo_array_crop_np, size_of_part_x, size_of_part_y)
                max_T_part = np.amax(thermo_array_crop_part)
                min_T_part = np.amin(thermo_array_crop_part)
                cropped_part_in_y_mm, cropped_part_in_x_mm = thermo_array_crop_part.shape  # cropped part call-out!! # x and y have been switched here so that the size will printed in the master file appropiately

                cropped_part_2_in_x_mm = cropped_part_in_x_mm / mm_in_x
                cropped_part_2_in_x_mm = round(cropped_part_2_in_x_mm, 3)
                cropped_part_2_in_y_mm = cropped_part_in_y_mm / mm_in_y
                cropped_part_2_in_y_mm = round(cropped_part_2_in_y_mm, 3)
                print('size of cropped part =', cropped_part_2_in_x_mm, '*', cropped_part_2_in_y_mm)
                # cropping is done. the callouts for the cropped tool and part array is found in comments

                # start with cropped tool
                # Mean value calculation
                mean_tool = np.mean(thermo_tool_crop_np)  # calculate mean value
                mean_tool = round(mean_tool, 3)
                print('Mean temperature: ', mean_tool)  # prints mean value in console

                #  e = pd.DataFrame(thermo_tool_crop_np)  # Black magic for pandas
                #  filepath = 'excel_converter3.xlsx'  # Gives the filepath
                #  e.to_excel(filepath, index=False)  # converts DataFrame to excel

                # Filename get
                filename = os.path.splitext(name)[0]  # split the file name from the original extension
                # 3D homogeneity calculation
                avg = np.average(thermo_tool_crop_np)
                abc = abs(avg - thermo_tool_crop_np)
                # print('3rd element on 56th dim: ', img[3, 56])
                # for i in range(h_tool_crop):
                #     globals()[f"list{i}"] = []
                #     for j in range(w_tool_crop):
                #         globals()[f"list{i}"].append(abs(avg - thermo_tool_crop_np[i, j]))
                #         if j == w_tool_crop - 1:
                #             globals()[f"list{i}"] = np.array(globals()[f"list{i}"])  # Converts from listo ndarray
                # abc = list0
                # for i in range(h_tool_crop):
                #     abc = np.vstack((abc, globals()[f"list{i + 1}"]))
                #     if i == h_tool_crop - 2:
                #         break

                middle = math.floor(w_tool_crop / 2)  # Defines where the middle of the picture is
                for i in range(h_tool_crop):
                    globals()[f"list_a{i}"] = []
                    middle_column1 = (
                    thermo_tool_crop_np[i, middle + 2])  # Looks shitty but this converts the middle 5 values to 1
                    middle_column1 = float(middle_column1)
                    middle_column2 = (thermo_tool_crop_np[i, middle + 1])
                    middle_column2 = float(middle_column2)
                    middle_column3 = (thermo_tool_crop_np[i, middle])
                    middle_column3 = float(middle_column3)
                    middle_column4 = (thermo_tool_crop_np[i, middle - 1])
                    middle_column4 = float(middle_column4)
                    middle_column5 = (thermo_tool_crop_np[i, middle - 2])
                    middle_column5 = float(middle_column5)
                    middle_column = ((
                                                 middle_column1 + middle_column2 + middle_column3 + middle_column4 + middle_column5) / 5)
                    middle_column = float(middle_column)
                    globals()[f"list_a{i}"].append(middle_column)
                    globals()[f"list_a{i}"] = np.array(globals()[f"list_a{i}"])  # Converts from list to ndarray
                abc2 = list_a0
                for i in range(h_tool_crop):
                    abc2 = np.vstack(
                        (abc2, globals()[f"list_a{i + 1}"]))  # stacking the 1 values to create the mean line
                    if i == h_tool_crop - 2:
                        break
                avg_line = np.average(abc2)  # gets the average value of the mean line
                for i in range(h_tool_crop):
                    globals()[f"list_b{i}"] = []
                    globals()[f"list_b{i}"].append(abs(avg_line - abc2[i]))  # error from the mean line to the average 5
                    globals()[f"list_b{i}"] = np.array(globals()[f"list_b{i}"])  # Converts from listo ndarray
                abc3 = list_b0
                for i in range(h_tool_crop):
                    abc3 = np.vstack((abc3, globals()[f"list_b{i + 1}"]))
                    if i == h_tool_crop - 2:
                        break
                homogeneity_line_sum = np.sum(abc3)
                homogeneity_line_sum = round(homogeneity_line_sum, 3)
                # Printing values
                print('Homogeneity line: ', homogeneity_line_sum)
                column_size_tool = len(abc[0])
                row_size_tool = len(abc)
                print('Size of 3D homogeneity matrix: ', column_size_tool, "*", row_size_tool)
                print('Homogeneity: ', np.sum(abc))
                homogeneity_sum = np.sum(abc)
                homogeneity_sum = round(homogeneity_sum, 3)
                print('Homogeneity reduced: ', homogeneity_sum)
                # 3D Homogeneity value divided with size of picture. Experimental
                mean_error = homogeneity_sum / (column_size_tool * row_size_tool)
                print('Homogeneity / pic size: ', mean_error)

                # Skewness and kurtosis for experimental. The 2D arrays need to be first flattened to get one value
                img_flat = np.concatenate(thermo_tool_crop_np).flat
                # img_flattened = pd.Series(img_flat)
                img_flattened_np = np.array(img_flat)
                st_dev_tool_1 = np.std(img_flattened_np)
                s = skew(img_flattened_np)
                print('Skewness: ', s)
                k = kurtosis(img_flattened_np)
                print('Kurtosis: ', k)

                # Homogeneity normalization - tool
                half_column_size_tool = column_size_tool / 2
                half_row_size_tool = row_size_tool / 2
                theoretical_inhomogeneity_average_tool = (max_T - 23) / 2
                theoretical_inhomogeneity_tool = theoretical_inhomogeneity_average_tool * row_size_tool * column_size_tool
                theoretical_inhomogeneity_tool = round(theoretical_inhomogeneity_tool, 3)
                homogeneity_percentage_tool = (
                            (1 - (homogeneity_sum / theoretical_inhomogeneity_tool)) * 100)  # TODO CHECK this

                # Insertion of calculated values into the previous empty phyton arrays
                meanarray_tool.insert(img_number,
                                      mean_tool)  # Inserts mean value into mean value array. img_number=index
                filenamearray_tool.insert(img_number, filename)  # Inserts name of file into filename array
                homoarray_tool.insert(img_number, homogeneity_sum)
                st_dev_tool.insert(img_number, st_dev_tool_1)
                skewarray_tool.insert(img_number, s)
                kurtarray_tool.insert(img_number, k)
                meanerror_array_tool.insert(img_number, mean_error)
                homogeneity_line_array_tool.insert(img_number, homogeneity_line_sum)
                w_tool_array.insert(img_number, cropped_tool_in_x_mm)
                h_tool_array.insert(img_number, cropped_tool_in_y_mm)
                homogeneity_percentage_tool_array.insert(img_number, homogeneity_percentage_tool)
                # Adds one to the counter

                # now for the part thermo_array_crop_part

                # Mean value calculation
                mean_part = np.mean(thermo_array_crop_part)  # calculate mean value
                mean_part = round(mean_part, 3)
                print('Mean temperature: ', mean_part)  # prints mean value in console

                #  e = pd.DataFrame(thermo_array_crop_part)  # Black magic for pandas
                #  filepath = 'excel_converter6.xlsx'  # Gives the filepath
                #  e.to_excel(filepath, index=False)  # converts DataFrame to excel

                # Filename get
                filename = os.path.splitext(name)[0]  # split the file name from the original extension
                # 3D homogeneity calculation
                avg_part = np.average(thermo_array_crop_part)
                # print('3rd element on 56th dim: ', img[3, 56])
                for i in range(size_of_part_y):
                    globals()[f"list_part{i}"] = []
                    for j in range(size_of_part_x):
                        globals()[f"list_part{i}"].append(abs(avg_part - thermo_array_crop_part[i, j]))
                        if j == size_of_part_x - 1:
                            globals()[f"list_part{i}"] = np.array(
                                globals()[f"list_part{i}"])  # Converts from listo ndarray
                abc_part1 = list_part0
                for i in range(size_of_part_y):
                    abc_part1 = np.vstack((abc_part1, globals()[f"list_part{i + 1}"]))
                    if i == size_of_part_y - 2:
                        break

                middle_part = math.floor(size_of_part_x / 2)  # Defines where the middle of the picture is
                for i in range(size_of_part_y):
                    globals()[f"list_part_b{i}"] = []
                    middle_column1_part = (thermo_array_crop_part[
                        i, middle_part + 2])  # Looks shitty but this converts the middle 5 values to 1
                    middle_column1_part = float(middle_column1_part)
                    middle_column2_part = (thermo_array_crop_part[i, middle_part + 1])
                    middle_column2_part = float(middle_column2_part)
                    middle_column3_part = (thermo_array_crop_part[i, middle_part])
                    middle_column3_part = float(middle_column3_part)
                    middle_column4_part = (thermo_array_crop_part[i, middle_part - 1])
                    middle_column4_part = float(middle_column4_part)
                    middle_column5_part = (thermo_array_crop_part[i, middle_part - 2])
                    middle_column5_part = float(middle_column5_part)
                    middle_column_part = ((
                                                      middle_column1_part + middle_column2_part + middle_column3_part + middle_column4_part + middle_column5_part) / 5)
                    middle_column_part = float(middle_column_part)
                    globals()[f"list_part_b{i}"].append(middle_column_part)
                    globals()[f"list_part_b{i}"] = np.array(
                        globals()[f"list_part_b{i}"])  # Converts from list to ndarray
                abc_part2 = list_part_b0
                for i in range(size_of_part_y):
                    abc_part2 = np.vstack(
                        (abc_part2, globals()[f"list_part_b{i + 1}"]))  # stacking the 1 values to create the mean line
                    if i == size_of_part_y - 2:
                        break
                avg_line_part = np.average(abc_part2)  # gets the average value of the mean line
                for i in range(size_of_part_y):
                    globals()[f"list_part_c{i}"] = []
                    globals()[f"list_part_c{i}"].append(
                        abs(avg_line_part - abc_part2[i]))  # error from the mean line to the average 5
                    globals()[f"list_part_c{i}"] = np.array(globals()[f"list_part_c{i}"])  # Converts from listo ndarray
                abc_part3 = list_part_c0
                for i in range(size_of_part_y):
                    abc_part3 = np.vstack((abc_part3, globals()[f"list_part_c{i + 1}"]))
                    if i == size_of_part_y - 2:
                        break
                homogeneity_line_sum_part = np.sum(abc_part3)
                homogeneity_line_sum_part = round(homogeneity_line_sum_part, 3)

                # If line graphs are necessary.
                # The script below changes the axis ticks, so that they are centered, and the distance is written on them. There might some rounding error, but it is good enough
                x_unmodified_ticks_tool = list(range(0, w_tool))
                x_unmodified_ticks_tool = np.array(x_unmodified_ticks_tool)
                x_modified_ticks_tool = [item / mm_in_x for item in x_unmodified_ticks_tool]
                x_modified_ticks_tool = [round(item, 1) for item in x_modified_ticks_tool]

                middle0 = x_modified_ticks_tool[0]
                middle1 = x_modified_ticks_tool[math.floor((len(x_modified_ticks_tool)) / 2)]
                middle2 = x_modified_ticks_tool[math.floor((len(x_modified_ticks_tool) / 4))]
                middle3 = x_modified_ticks_tool[math.floor((len(x_modified_ticks_tool) / 8))]
                middle4 = x_modified_ticks_tool[
                    math.floor((len(x_modified_ticks_tool) / 4) + (len(x_modified_ticks_tool) / 8))]
                middle5 = x_modified_ticks_tool[
                    math.floor((len(x_modified_ticks_tool) / 2) + (len(x_modified_ticks_tool) / 8))]
                middle6 = x_modified_ticks_tool[
                    math.floor((len(x_modified_ticks_tool) / 2) + (len(x_modified_ticks_tool) / 4))]
                middle7 = x_modified_ticks_tool[math.floor(
                    (len(x_modified_ticks_tool) / 2) + (len(x_modified_ticks_tool) / 4) + (
                            len(x_modified_ticks_tool) / 8))]
                middle8 = x_modified_ticks_tool[-1]
                middle_list = []
                middle_list.extend([middle0, middle3, middle2, middle4, middle1, middle5, middle6, middle7, middle8])

                middleb0 = x_unmodified_ticks_tool[0]
                middleb1 = x_unmodified_ticks_tool[math.floor((len(x_unmodified_ticks_tool)) / 2)]
                middleb2 = x_unmodified_ticks_tool[math.floor((len(x_unmodified_ticks_tool) / 4))]
                middleb3 = x_unmodified_ticks_tool[math.floor((len(x_unmodified_ticks_tool) / 8))]
                middleb4 = x_unmodified_ticks_tool[
                    math.floor((len(x_unmodified_ticks_tool) / 4) + (len(x_unmodified_ticks_tool) / 8))]
                middleb5 = x_unmodified_ticks_tool[
                    math.floor((len(x_unmodified_ticks_tool) / 2) + (len(x_unmodified_ticks_tool) / 8))]
                middleb6 = x_unmodified_ticks_tool[
                    math.floor((len(x_unmodified_ticks_tool) / 2) + (len(x_unmodified_ticks_tool) / 4))]
                middleb7 = x_unmodified_ticks_tool[math.floor(
                    (len(x_unmodified_ticks_tool) / 2) + (len(x_unmodified_ticks_tool) / 4) + (
                            len(x_unmodified_ticks_tool) / 8))]
                middleb8 = x_unmodified_ticks_tool[-1]
                middle_list_b = []
                middle_list_b.extend(
                    [middleb0, middleb3, middleb2, middleb4, middleb1, middleb5, middleb6, middleb7, middleb8])
                plt.xticks((middle_list_b), (middle_list))
                plt.title("Line graph, cropped tool")
                plt.xlabel("Central Line [mm]")
                plt.ylabel("Temperature")
                length_of_homoline, homoline_shape = abc3.shape
                x2 = np.arange(1, length_of_homoline + 1)
                x3 = np.array(x2)
                x = x3.reshape(-1, 1)

                abc4 = abc2.reshape(-1, 1)

                plt.hlines(avg_line, 0, length_of_homoline, color='black', zorder=2, label='Average line')
                plt.plot(x, abc4, color="red", zorder=1, label='Average temperature values')
                plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
                plt.savefig('plots/homline/' + str(filename) + "_hom_line" + ".png", bbox_inches='tight')
                plt.clf()

                # If line graphs are necessary.
                # The script below changes the axis ticks, so that they are centered, and the distance is written on them. There might some rounding error, but it is good enough
                x_unmodified_ticks_part = list(range(0, size_of_part_x))
                x_unmodified_ticks_part = np.array(x_unmodified_ticks_part)
                x_modified_ticks_part = [item / mm_in_x for item in x_unmodified_ticks_part]
                x_modified_ticks_part = [round(item, 1) for item in x_modified_ticks_part]

                middle0 = x_modified_ticks_part[0]
                middle1 = x_modified_ticks_part[math.floor((len(x_modified_ticks_part)) / 2)]
                middle2 = x_modified_ticks_part[math.floor((len(x_modified_ticks_part) / 4))]
                middle3 = x_modified_ticks_part[math.floor((len(x_modified_ticks_part) / 8))]
                middle4 = x_modified_ticks_part[
                    math.floor((len(x_modified_ticks_part) / 4) + (len(x_modified_ticks_part) / 8))]
                middle5 = x_modified_ticks_part[
                    math.floor((len(x_modified_ticks_part) / 2) + (len(x_modified_ticks_part) / 8))]
                middle6 = x_modified_ticks_part[
                    math.floor((len(x_modified_ticks_part) / 2) + (len(x_modified_ticks_part) / 4))]
                middle7 = x_modified_ticks_part[math.floor(
                    (len(x_modified_ticks_part) / 2) + (len(x_modified_ticks_part) / 4) + (
                            len(x_modified_ticks_part) / 8))]
                middle8 = x_modified_ticks_part[-1]
                middle_list = []
                middle_list.extend([middle0, middle3, middle2, middle4, middle1, middle5, middle6, middle7, middle8])

                middleb0 = x_unmodified_ticks_part[0]
                middleb1 = x_unmodified_ticks_part[math.floor((len(x_unmodified_ticks_part)) / 2)]
                middleb2 = x_unmodified_ticks_part[math.floor((len(x_unmodified_ticks_part) / 4))]
                middleb3 = x_unmodified_ticks_part[math.floor((len(x_unmodified_ticks_part) / 8))]
                middleb4 = x_unmodified_ticks_part[
                    math.floor((len(x_unmodified_ticks_part) / 4) + (len(x_unmodified_ticks_part) / 8))]
                middleb5 = x_unmodified_ticks_part[
                    math.floor((len(x_unmodified_ticks_part) / 2) + (len(x_unmodified_ticks_part) / 8))]
                middleb6 = x_unmodified_ticks_part[
                    math.floor((len(x_unmodified_ticks_part) / 2) + (len(x_unmodified_ticks_part) / 4))]
                middleb7 = x_unmodified_ticks_part[math.floor(
                    (len(x_unmodified_ticks_part) / 2) + (len(x_unmodified_ticks_part) / 4) + (
                            len(x_unmodified_ticks_part) / 8))]
                middleb8 = x_unmodified_ticks_part[-1]
                middle_list_b = []
                middle_list_b.extend(
                    [middleb0, middleb3, middleb2, middleb4, middleb1, middleb5, middleb6, middleb7, middleb8])
                plt.xticks((middle_list_b), (middle_list))
                plt.title("Line graph, cropped part")
                plt.xlabel("Central Line [mm]")
                plt.ylabel("Temperature")
                length_of_homoline_part, homoline_shape_part = abc_part3.shape
                x2 = np.arange(1, length_of_homoline_part + 1)
                x3 = np.array(x2)
                x = x3.reshape(-1, 1)

                abc_part4 = abc_part2.reshape(-1, 1)

                plt.hlines(avg_line_part, 0, length_of_homoline_part, color='black', zorder=2, label='Average line')
                plt.plot(x, abc_part4, color="red", zorder=1, label='Average temperature values')
                plt.legend(bbox_to_anchor=(1.04, 0.5),
                           loc="center left")  # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
                plt.savefig('plots/homline_part/' + str(filename) + "_hom_line_part" + ".png", bbox_inches='tight')
                plt.clf()
                middle_list.clear()
                middle_list_b.clear()

                # Printing values
                print('Homogeneity line: ', homogeneity_line_sum_part)
                column_size = len(abc_part1[0])
                row_size = len(abc_part1)
                print('Size of 3D homogeneity matrix: ', column_size, "*", row_size)
                print('Homogeneity: ', np.sum(abc_part1))
                homogeneity_sum_part = np.sum(abc_part1)
                homogeneity_sum_part = round(homogeneity_sum_part, 3)
                print('Homogeneity reduced: ', homogeneity_sum_part)
                # 3D Homogeneity value divided with size of picture. Experimental
                mean_error_part = homogeneity_sum_part / (size_of_part_x * size_of_part_y)
                print('Homogeneity / pic size: ', mean_error_part)

                # Skewness and kurtosis for experimental. The 2D arrays need to be first flattened to get one value
                img_flat_part = np.concatenate(thermo_array_crop_part).flat
                # img_flattened = pd.Series(img_flat)
                img_flattened_np_part = np.array(img_flat_part)
                st_dev_part_1 = np.std(img_flattened_np_part)
                s_part = skew(img_flattened_np_part)
                print('Skewness: ', s)
                k_part = kurtosis(img_flattened_np_part)
                print('Kurtosis: ', k)

                # Show less homogene areas
                part_area = thermo_array_crop_part.copy()
                threshold = set_temperature - threshold_limit
                threshold_2 = set_temperature + threshold_limit
                part_area[part_area < threshold] = 0
                part_area[part_area > threshold_2] = 0  # change low T values to 0

                # Homogeneity normalization - part
                half_column_size_part = size_of_part_x / 2
                half_row_size_part = size_of_part_y / 2
                theoretical_inhomogeneity_average_part = (max_T_part - 23) / 2
                theoretical_inhomogeneity_part = theoretical_inhomogeneity_average_part * size_of_part_y * size_of_part_x
                theoretical_inhomogeneity_part = round(theoretical_inhomogeneity_part, 3)
                homogeneity_percentage_part = ((1 - (homogeneity_sum_part / theoretical_inhomogeneity_part)) * 100)

                # Warning system for part inhomogeneity
                zeros_column_part = (part_area == 0).sum(0)
                total_zeroes_part = np.sum(zeros_column_part)
                total_part_size = size_of_part_y * size_of_part_x
                percentage_zero = (total_zeroes_part / total_part_size) * 100
                if percentage_zero < percentage_for_ok_not_ok:
                    warning_zero_part.insert(img_number, "ok")
                else:
                    warning_zero_part.insert(img_number, "not ok")

                # Insertion of calculated values into the previous empty phyton arrays
                meanarray_part.insert(img_number,
                                      mean_part)  # Inserts mean value into mean value array. img_number=index
                filenamearray_part.insert(img_number, filename)  # Inserts name of file into filename array
                homoarray_part.insert(img_number, homogeneity_sum_part)
                st_dev_part.insert(img_number, st_dev_part_1)
                skewarray_part.insert(img_number, s_part)
                kurtarray_part.insert(img_number, k_part)
                meanerror_array_part.insert(img_number, mean_error_part)
                homogeneity_line_array_part.insert(img_number, homogeneity_line_sum_part)
                w_part_array.insert(img_number, cropped_part_2_in_x_mm)
                h_part_array.insert(img_number, cropped_part_2_in_y_mm)
                homogeneity_percentage_part_array.insert(img_number, homogeneity_percentage_part)
                img_number += 1
                print("--- %s seconds ---" % (time.time() - start_time))

                import math
                import os
                from matplotlib import pyplot as plt
                from scipy.stats import skew, kurtosis
                import numpy as np
                import pandas as pd
                import time

                # points of interests for the temperature measurements at different positions
                # middle
                part_middle_poi_x = math.floor(w_tool_crop / 2)
                part_middle_poi_x_pos = part_middle_poi_x + (cross_size / 2)
                part_middle_poi_x_draw_1 = part_middle_poi_x - cross_size
                part_middle_poi_x_draw_2 = part_middle_poi_x + cross_size
                part_middle_poi_y = math.floor(h_tool_crop / 2)
                part_middle_poi_y_pos = part_middle_poi_y - cross_size
                part_middle_poi_y_draw_1 = part_middle_poi_y - cross_size
                part_middle_poi_y_draw_2 = part_middle_poi_y + cross_size
                part_middle_poi_T = thermo_tool_crop_np[part_middle_poi_y, part_middle_poi_x]

                part_bottomleft_poi_x = math.floor(part_middle_poi_x / 2)
                part_bottomleft_poi_x_pos = part_bottomleft_poi_x + (cross_size / 2)
                part_bottomleft_poi_x_draw_1 = part_bottomleft_poi_x - cross_size
                part_bottomleft_poi_x_draw_2 = part_bottomleft_poi_x + cross_size
                part_bottomleft_poi_y = math.floor(part_middle_poi_y / 2)
                part_bottomleft_poi_y_pos = part_bottomleft_poi_y - cross_size
                part_bottomleft_poi_y_draw_1 = part_bottomleft_poi_y - cross_size
                part_bottomleft_poi_y_draw_2 = part_bottomleft_poi_y + cross_size
                part_bottomleft_poi_T = thermo_tool_crop_np[part_bottomleft_poi_y, part_bottomleft_poi_x]

                part_bottomright_poi_x = math.floor((part_middle_poi_x / 2) + part_middle_poi_x)
                part_bottomright_poi_x_pos = part_bottomright_poi_x + (cross_size / 2)
                part_bottomright_poi_x_draw_1 = part_bottomright_poi_x - cross_size
                part_bottomright_poi_x_draw_2 = part_bottomright_poi_x + cross_size
                part_bottomright_poi_y = math.floor(part_middle_poi_y / 2)
                part_bottomright_poi_y_pos = part_bottomright_poi_y - cross_size
                part_bottomright_poi_y_draw_1 = part_bottomright_poi_y - cross_size
                part_bottomright_poi_y_draw_2 = part_bottomright_poi_y + cross_size
                part_bottomright_poi_T = thermo_tool_crop_np[part_bottomright_poi_y, part_bottomright_poi_x]

                part_topleft_poi_x = math.floor(part_middle_poi_x / 2)
                part_topleft_poi_x_pos = part_topleft_poi_x + (cross_size / 2)
                part_topleft_poi_x_draw_1 = part_topleft_poi_x - cross_size
                part_topleft_poi_x_draw_2 = part_topleft_poi_x + cross_size
                part_topleft_poi_y = math.floor(part_middle_poi_y / 2 + part_middle_poi_y)
                part_topleft_poi_y_pos = part_topleft_poi_y - cross_size
                part_topleft_poi_y_draw_1 = part_topleft_poi_y - cross_size
                part_topleft_poi_y_draw_2 = part_topleft_poi_y + cross_size
                part_topleft_poi_T = thermo_tool_crop_np[part_topleft_poi_y, part_topleft_poi_x]

                part_topright_poi_x = math.floor(part_middle_poi_x / 2 + part_middle_poi_x)
                part_topright_poi_x_pos = part_topright_poi_x + (cross_size / 2)
                part_topright_poi_x_draw_1 = part_topright_poi_x - cross_size
                part_topright_poi_x_draw_2 = part_topright_poi_x + cross_size
                part_topright_poi_y = math.floor(part_middle_poi_y / 2 + part_middle_poi_y)
                part_topright_poi_y_pos = part_topright_poi_y - cross_size
                part_topright_poi_y_draw_1 = part_topright_poi_y - cross_size
                part_topright_poi_y_draw_2 = part_topright_poi_y + cross_size
                part_topright_poi_T = thermo_tool_crop_np[part_topright_poi_y, part_topright_poi_x]

                part_middletop_poi_x = math.floor(part_middle_poi_x)
                part_middletop_poi_x_pos = part_middletop_poi_x + (cross_size / 2)
                part_middletop_poi_x_draw_1 = part_middletop_poi_x - cross_size
                part_middletop_poi_x_draw_2 = part_middletop_poi_x + cross_size
                part_middletop_poi_y = math.floor(part_middle_poi_y / 2 + part_middle_poi_y)
                part_middletop_poi_y_pos = part_middletop_poi_y - cross_size
                part_middletop_poi_y_draw_1 = part_middletop_poi_y - cross_size
                part_middletop_poi_y_draw_2 = part_middletop_poi_y + cross_size
                part_middletop_poi_T = thermo_tool_crop_np[part_middletop_poi_y, part_middletop_poi_x]

                part_middlebottom_poi_x = math.floor(part_middle_poi_x)
                part_middlebottom_poi_x_pos = part_middlebottom_poi_x + (cross_size / 2)
                part_middlebottom_poi_x_draw_1 = part_middlebottom_poi_x - cross_size
                part_middlebottom_poi_x_draw_2 = part_middlebottom_poi_x + cross_size
                part_middlebottom_poi_y = math.floor(part_middle_poi_y - part_middle_poi_y / 2)
                part_middlebottom_poi_y_pos = part_middlebottom_poi_y - cross_size
                part_middlebottom_poi_y_draw_1 = part_middlebottom_poi_y - cross_size
                part_middlebottom_poi_y_draw_2 = part_middlebottom_poi_y + cross_size
                part_middlebottom_poi_T = thermo_tool_crop_np[part_middlebottom_poi_y, part_middlebottom_poi_x]

                part_centerleft_poi_x = math.floor((part_middle_poi_x + part_topleft_poi_x) / 2)
                part_centerleft_poi_x_pos = part_centerleft_poi_x - (cross_size / 2) + (cross_size)
                part_centerleft_poi_x_draw_1 = part_centerleft_poi_x - cross_size
                part_centerleft_poi_x_draw_2 = part_centerleft_poi_x + cross_size
                part_centerleft_poi_y = math.floor(part_middle_poi_y)
                part_centerleft_poi_y_pos = part_centerleft_poi_y - cross_size
                part_centerleft_poi_y_draw_1 = part_centerleft_poi_y - cross_size
                part_centerleft_poi_y_draw_2 = part_centerleft_poi_y + cross_size
                part_centerleft_poi_T = thermo_tool_crop_np[part_centerleft_poi_y, part_centerleft_poi_x]

                part_centerright_poi_x = math.floor((part_middle_poi_x + part_topright_poi_x) / 2)
                part_centerright_poi_x_pos = part_centerright_poi_x + 10
                part_centerright_poi_x_draw_1 = part_centerright_poi_x - cross_size
                part_centerright_poi_x_draw_2 = part_centerright_poi_x + cross_size
                part_centerright_poi_y = math.floor(part_middle_poi_y)
                part_centerright_poi_y_pos = part_centerright_poi_y - cross_size
                part_centerright_poi_y_draw_1 = part_centerright_poi_y - cross_size
                part_centerright_poi_y_draw_2 = part_centerright_poi_y + cross_size
                part_centerright_poi_T = thermo_tool_crop_np[part_centerright_poi_y, part_centerright_poi_x]

                # show image of the tool
                plt.pcolormesh(thermo_array_crop_full_tool, cmap="viridis", vmin=min_T_full_tool, vmax=max_T_full_tool)
                plt.axis('scaled')
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
                plt.title("Plot 2D array, full tool")
                plt.grid('major')

                # The script below changes the axis ticks, so that they are centered, and the distance is written on them. There might some rounding error, but it is good enough
                x_axis(w_tool)
                y_axis(h_tool)

                plt.colorbar()
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(12, 8)  # here the size can be changed. original: 8, 6
                plt.savefig('plots/full_tool/' + str(filename) + "_full_tool" + ".png", bbox_inches='tight',
                            dpi=200)  # dpi 200 makes it bigger
                # plt.show()
                plt.clf()

                # show image of the cropped tool
                plt.pcolormesh(thermo_tool_crop_np, cmap="viridis", vmin=0, vmax=max_T)
                plt.axis('scaled')
                plt.title("Plot 2D array, cropped tool")
                plt.grid('major')
                plt.colorbar()
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(12, 8)
                plt.gca().invert_yaxis()
                plt.savefig('plots/cropped_tool/' + str(filename) + "_cropped_tool" + ".png", bbox_inches='tight',
                            dpi=200)
                # plt.show()
                plt.clf()

                # show image with mm in labels of the cropped tool
                plt.pcolormesh(thermo_tool_crop_np, cmap="viridis", vmin=0, vmax=max_T)
                plt.axis('scaled')
                plt.title("Plot 2D array, cropped tool")

                x_axis(w_tool_crop)
                y_axis(h_tool_crop)

                plt.grid('major')
                plt.colorbar()
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(12, 8)
                plt.savefig('plots/cropped_tool_mm/' + str(filename) + "_cropped_tool_mm" + ".png", bbox_inches='tight',
                            dpi=200)
                # plt.show()
                plt.clf()

                # show image of the cropped with temperature tool - not used for now. The axis scale is with pixels. its ok to use it, but unncessary
                #  plt.pcolormesh(thermo_tool_crop_np, cmap="viridis", vmin=min_T, vmax=max_T) # vmax=max_T
                #  plt.axis('scaled')
                #  plt.title("Plot 2D array, cropped tool")
                #  plt.colorbar()
                #  figure = plt.gcf()  # get current figure
                #  figure.set_size_inches(12, 8)
                #  plt.savefig('plots/cropped_tool_with_temp/' + str(filename) + "_cropped_tool_with_temp" + ".png", bbox_inches='tight', dpi = 200)
                #  plt.show()
                #  plt.clf()

                # show image of the cropped with temperature and mm tool
                plt.pcolormesh(thermo_tool_crop_np, cmap="viridis", vmin=min_T, vmax=max_T)  # vmax=max_T
                plt.axis('scaled')

                x_axis(w_tool_crop)
                y_axis(h_tool_crop)
                plt.grid('major')

                # drawing for the POI
                plt.text(part_middle_poi_x_pos, part_middle_poi_y_pos, f"{part_middle_poi_T}°C", fontsize=8)
                plt.hlines(part_middle_poi_y, part_middle_poi_x_draw_1, part_middle_poi_x_draw_2, colors='black')
                plt.vlines(part_middle_poi_x, part_middle_poi_y_draw_1, part_middle_poi_y_draw_2, colors='black')

                plt.text(part_bottomleft_poi_x_pos, part_bottomleft_poi_y_pos, f"{part_bottomleft_poi_T}°C", fontsize=8)
                plt.hlines(part_bottomleft_poi_y, part_bottomleft_poi_x_draw_1, part_bottomleft_poi_x_draw_2,
                           colors='black')
                plt.vlines(part_bottomleft_poi_x, part_bottomleft_poi_y_draw_1, part_bottomleft_poi_y_draw_2,
                           colors='black')

                plt.text(part_bottomright_poi_x_pos, part_bottomright_poi_y_pos, f"{part_bottomright_poi_T}°C",
                         fontsize=8)
                plt.hlines(part_bottomright_poi_y, part_bottomright_poi_x_draw_1, part_bottomright_poi_x_draw_2,
                           colors='black')
                plt.vlines(part_bottomright_poi_x, part_bottomright_poi_y_draw_1, part_bottomright_poi_y_draw_2,
                           colors='black')

                plt.text(part_topleft_poi_x_pos, part_topleft_poi_y_pos, f"{part_topleft_poi_T}°C", fontsize=8)
                plt.hlines(part_topleft_poi_y, part_topleft_poi_x_draw_1, part_topleft_poi_x_draw_2, colors='black')
                plt.vlines(part_topleft_poi_x, part_topleft_poi_y_draw_1, part_topleft_poi_y_draw_2, colors='black')

                plt.text(part_topright_poi_x_pos, part_topright_poi_y_pos, f"{part_topright_poi_T}°C", fontsize=8)
                plt.hlines(part_topright_poi_y, part_topright_poi_x_draw_1, part_topright_poi_x_draw_2, colors='black')
                plt.vlines(part_topright_poi_x, part_topright_poi_y_draw_1, part_topright_poi_y_draw_2, colors='black')

                plt.text(part_middletop_poi_x_pos, part_middletop_poi_y_pos, f"{part_middletop_poi_T}°C", fontsize=8)
                plt.hlines(part_middletop_poi_y, part_middletop_poi_x_draw_1, part_middletop_poi_x_draw_2,
                           colors='black')
                plt.vlines(part_middletop_poi_x, part_middletop_poi_y_draw_1, part_middletop_poi_y_draw_2,
                           colors='black')

                plt.text(part_middlebottom_poi_x_pos, part_middlebottom_poi_y_pos, f"{part_middlebottom_poi_T}°C",
                         fontsize=8)
                plt.hlines(part_middlebottom_poi_y, part_middlebottom_poi_x_draw_1, part_middlebottom_poi_x_draw_2,
                           colors='black')
                plt.vlines(part_middlebottom_poi_x, part_middlebottom_poi_y_draw_1, part_middlebottom_poi_y_draw_2,
                           colors='black')

                plt.text(part_centerleft_poi_x_pos, part_centerleft_poi_y_pos, f"{part_centerleft_poi_T}°C", fontsize=8)
                plt.hlines(part_centerleft_poi_y, part_centerleft_poi_x_draw_1, part_centerleft_poi_x_draw_2,
                           colors='black')
                plt.vlines(part_centerleft_poi_x, part_centerleft_poi_y_draw_1, part_centerleft_poi_y_draw_2,
                           colors='black')

                plt.text(part_centerright_poi_x_pos, part_centerright_poi_y_pos, f"{part_centerright_poi_T}°C",
                         fontsize=8)
                plt.hlines(part_centerright_poi_y, part_centerright_poi_x_draw_1, part_centerright_poi_x_draw_2,
                           colors='black')
                plt.vlines(part_centerright_poi_x, part_centerright_poi_y_draw_1, part_centerright_poi_y_draw_2,
                           colors='black')

                plt.grid('major')
                plt.title("Plot 2D array, cropped tool of "+str(filename))
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
                plt.colorbar()
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(12, 8)
                plt.savefig('plots/cropped_tool_with_temp_mm/' + str(filename) + "_cropped_tool_with_temp" + ".png",
                            bbox_inches='tight', dpi=200)
                plt.savefig('plots/cropped_tool_with_temp_mm/' + "cropped_tool_with_temp_demo" + ".png",
                            bbox_inches='tight', dpi=600)
                #  plt.show()
                plt.clf()

                # show image of the part
                plt.pcolormesh(thermo_array_crop_part, cmap="viridis", vmin=min_T_part, vmax=max_T_part)
                plt.axis('scaled')
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
                plt.title("Plot 2D array, cropped part of" + str(filename))

                x_axis(size_of_part_x)
                y_axis(size_of_part_y)

                plt.grid('major')
                plt.colorbar()
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(8, 6)
                plt.savefig('plots/part/' + str(filename) + "_part" + ".png", bbox_inches='tight', dpi=200)
                #   plt.show()
                plt.clf()

                # show tresholded area
                plt.pcolormesh(part_area, cmap="viridis", vmin=0, vmax=max_T_part)
                plt.axis('scaled')
                plt.title("Plot 2D array, thresholded")

                x_axis(size_of_part_x)
                y_axis(size_of_part_y)

                if percentage_zero < percentage_for_ok_not_ok:
                    plt.text((size_of_part_x / 2), -20, "OK", fontsize=12)
                else:
                    plt.text((size_of_part_x / 2) - 6, -20, "not OK", fontsize=12)
                plt.colorbar()
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
                plt.grid('major')
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(12, 8)
                plt.savefig('plots/threshold/' + str(filename) + "_threshold" + ".png", bbox_inches='tight', dpi=200)
                # plt.show()
                plt.clf()

                # Histogram without distribution
                if w_zero_int == part_bottom_right_corner_x:
                    pass
                else:
                    img_flattened_np_pd = pd.Series(img_flattened_np_part)
                    img_flattened_np_pd.plot.hist(grid=True, bins=max_T_part, width=0.8,
                                                  color='#607c8e', density=False, histtype='bar')
                    plt.title('T histogram of the part ' + str(filename))
                    plt.xlabel('Temperature [°C]')
                    plt.ylabel('Frequency')
                    plt.grid(axis='y', alpha=0.75)
                    plt.savefig('plots/histograms/' + str(filename) + "_histo" + ".png", bbox_inches='tight')
                    plt.clf()

                    # Histogram with distribution: https://danielhnyk.cz/fitting-distribution-histogram-using-python/
                    img_flattened_np_pd = pd.Series(img_flattened_np_part)
                    img_flattened_np_pd.plot.hist(grid=True, bins=max_T_part, width=0.8,
                                                  color='#607c8e', density=True, histtype='bar')

                    xt = plt.xticks()[0]
                    xmin, xmax = min(xt), max(xt)
                    lnspc = np.linspace(xmin, xmax, len(img_flattened_np_part))
                    m, s = scipy.stats.norm.fit(img_flattened_np_part)  # get mean and standard deviation
                    pdf_g = scipy.stats.norm.pdf(lnspc, m, s)  # now get theoretical values in our interval
                    plt.plot(lnspc, pdf_g, label="Norm")  # plot it
                    m_r = round(m, 2)
                    s_r = round(s, 2)
                    plt.title(
                        'T histogram of the part ' + str(filename) + ' Mean:' + str(m_r) + ' Deviation:' + str(s_r))
                    plt.xlabel('Temperature [°C]')
                    plt.ylabel('Frequency')
                    plt.grid(axis='y', alpha=0.75)

                    plt.savefig('plots/histograms_with_distributions/' + str(filename) + "_histo_dist" + ".png",
                                bbox_inches='tight')
                    plt.clf()

                ### Area-wise measurements

                ### Area 1
                h_tool_for_part_area_drawing, w_tool_for_part_area_drawing = thermo_array_crop_np.shape
                cordinate_area1_tl_row = h_tool_for_part_area_drawing / 2 - size_of_part_y / 2
                cordinate_area1_tl_column = w_tool_for_part_area_drawing / 2 - size_of_part_x / 2
                cordinate_area_1_br_row = cordinate_area1_tl_row + size_of_part_y
                cordinate_area_1_br_column = cordinate_area1_tl_column + size_of_part_x

                # Drawing
                plt.pcolormesh(thermo_tool_crop_np, cmap="viridis", vmin=min_T_full_tool, vmax=max_T_full_tool)
                plt.axis('scaled')

                x_axis(w_tool_crop)  # TODO check this
                y_axis(h_tool_crop)

                area1_drawing = part_middle_poi_x - cross_size
                area1_draw = part_middle_poi_x + cross_size
                #  plt.text(part_centerright_poi_x_pos, part_centerright_poi_y_pos, f"{part_centerright_poi_T}°C", fontsize=8)

                plt.hlines(cordinate_area1_tl_row, cordinate_area1_tl_column, cordinate_area_1_br_column,
                           colors='black', linewidth=0.5)  # first horizontal line
                plt.hlines(cordinate_area_1_br_row, cordinate_area1_tl_column, cordinate_area_1_br_column,
                           colors='black', linewidth=0.5)
                plt.vlines(cordinate_area1_tl_column, cordinate_area1_tl_row, cordinate_area_1_br_row, colors='black',
                           linewidth=0.5)
                plt.vlines(cordinate_area_1_br_column, cordinate_area1_tl_row, cordinate_area_1_br_row, colors='black',
                           linewidth=0.5)
                plt.grid('major')
                plt.title("Plot 2D array, full tool, divided by areas")
                plt.colorbar()
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(12, 8)
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
                plt.savefig('plots/full_tool_by_area/' + str(filename) + "_full_tool_by_area" + ".png",
                            bbox_inches='tight', dpi=200)
                plt.savefig('plots/dashboard/full_tool_by_area/' + "full_tool_by_area" + ".png",
                            bbox_inches='tight', dpi=200)
                #  plt.show()
                plt.clf()

                # show image of the part with dashed line
                plt.pcolormesh(thermo_array_crop_part, cmap="viridis", vmin=min_T_part, vmax=max_T_part, zorder=1)
                plt.axis('scaled')
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
                plt.title("Plot 2D array, cropped part of " + str(filename))
                part_middle_for_line = size_of_part_x / 2
                percentage_zero_reduced = round(percentage_zero, 2)
                x_axis(size_of_part_x)
                y_axis(size_of_part_y)
                #   plt.vlines(part_middle_for_line, 0, size_of_part_y, colors='black', zorder=2, linestyles='dashed')
                if percentage_zero < percentage_for_ok_not_ok:
                    plt.text((size_of_part_x / 2) - 10, -15, "OK:", fontsize=12)
                    plt.text((size_of_part_x / 2), -15, str(percentage_zero_reduced) + '%', fontsize=12,
                             color='green')  # was +0 +10 before
                else:
                    plt.text((size_of_part_x / 2) - 19, -15, "Not OK:", fontsize=12)  # set middle, write how much %
                    plt.text((size_of_part_x / 2) + 3, -15, str(percentage_zero_reduced) + '%', fontsize=12,
                             color='red')

                part_area_part_threshold_10_colder = thermo_array_crop_part.copy()
                threshold_2_part_threshold_10_colder = set_temperature - threshold_limit
                part_area_part_threshold_10_colder[
                    part_area_part_threshold_10_colder > threshold_2_part_threshold_10_colder] = 0
                part_area_part_threshold_10_colder = np.where(np.isnan(part_area_part_threshold_10_colder), 0,
                                                              part_area_part_threshold_10_colder)
                highlight = (part_area_part_threshold_10_colder < 1)
                # highlight = np.ma.masked_less(highlight, 1)
                x_part_linear = np.linspace(0, size_of_part_x - 1, size_of_part_x)
                y_part_linear = np.linspace(0, size_of_part_y - 1, size_of_part_y)
                plot_time_threshold_start = time.time()
                for i in range(len(y_part_linear)):
                    for j in range(len(x_part_linear)):
                        if highlight[i, j] == True:

                            if j == 0:
                                pass
                            else:
                                if highlight[
                                    i, j - 1] == False:  # the script goes through cell by cell. if there is a true value, it checks around the area if there is a false. if yes, a line is drawn
                                    plt.vlines(j, i, i + 1, colors='#00D8FF', zorder=2, linewidth=0.25)

                            if j > len(x_part_linear) - 2:
                                pass
                            else:
                                if highlight[i, j + 1] == False:
                                    plt.vlines(j + 1, i, i + 1, colors='#00D8FF', zorder=2, linewidth=0.25)

                            if i == 0:
                                pass
                            else:
                                if highlight[i - 1, j] == False:
                                    plt.hlines(i, j, j + 1, colors='#00D8FF', zorder=2, linewidth=0.25)

                            if i > len(y_part_linear) - 2:
                                pass
                            else:
                                if highlight[i + 1, j] == False:
                                    plt.hlines(i + 1, j, j + 1, colors='#00D8FF', zorder=2, linewidth=0.25)

                part_area_part_threshold_10_hotter = thermo_array_crop_part.copy()
                threshold_2_part_threshold_10_hotter = set_temperature + threshold_limit
                part_area_part_threshold_10_hotter[
                    part_area_part_threshold_10_hotter < threshold_2_part_threshold_10_hotter] = 0
                ### drawing qwert
                part_area_part_threshold_10_hotter[np.isnan(part_area_part_threshold_10_hotter)] = 0
                highlight = (part_area_part_threshold_10_hotter < 1)
                # highlight = np.ma.masked_less(highlight, 1)
                x_part_linear = np.linspace(0, size_of_part_x - 1, size_of_part_x)
                y_part_linear = np.linspace(0, size_of_part_y - 1, size_of_part_y)

                for i in range(len(y_part_linear)):
                    for j in range(len(x_part_linear)):
                        if highlight[i, j] == True:

                            if j == 0:
                                pass
                            else:
                                if highlight[
                                    i, j - 1] == False:  # the script goes through cell by cell. if there is a true value, it checks around the area if there is a false. if yes, a line is drawn
                                    plt.vlines(j, i, i + 1, colors='#FF7C00', zorder=2, linewidth=0.25)

                            if j > len(x_part_linear) - 2:
                                pass
                            else:
                                if highlight[i, j + 1] == False:
                                    plt.vlines(j + 1, i, i + 1, colors='#FF7C00', zorder=2, linewidth=0.25)

                            if i == 0:
                                pass
                            else:
                                if highlight[i - 1, j] == False:
                                    plt.hlines(i, j, j + 1, colors='#FF7C00', zorder=2, linewidth=0.25)

                            if i > len(y_part_linear) - 2:
                                pass
                            else:
                                if highlight[i + 1, j] == False:
                                    plt.hlines(i + 1, j, j + 1, colors='#FF7C00', zorder=2, linewidth=0.25)

                part_area_part_threshold_15_hotter = thermo_array_crop_part.copy()
                threshold_2_part_threshold_15_hotter = set_temperature + threshold_limit + 5
                part_area_part_threshold_15_hotter[
                    part_area_part_threshold_15_hotter > threshold_2_part_threshold_15_hotter] = 0
                ### drawing qwert
                part_area_part_threshold_15_hotter[np.isnan(part_area_part_threshold_15_hotter)] = avg
                highlight = (part_area_part_threshold_15_hotter < 1)
                # highlight = np.ma.masked_less(highlight, 1)
                x_part_linear = np.linspace(0, size_of_part_x - 1, size_of_part_x)
                y_part_linear = np.linspace(0, size_of_part_y - 1, size_of_part_y)

                for i in range(len(y_part_linear)):
                    for j in range(len(x_part_linear)):
                        if highlight[i, j] == True:

                            if j == 0:
                                pass
                            else:
                                if highlight[
                                    i, j - 1] == False:  # the script goes through cell by cell. if there is a true value, it checks around the area if there is a false. if yes, a line is drawn
                                    plt.vlines(j, i, i + 1, colors='#FF0000', zorder=2, linewidth=0.25)

                            if j > len(x_part_linear) - 2:
                                pass
                            else:
                                if highlight[i, j + 1] == False:
                                    plt.vlines(j + 1, i, i + 1, colors='#FF0000', zorder=2, linewidth=0.25)

                            if i == 0:
                                pass
                            else:
                                if highlight[i - 1, j] == False:
                                    plt.hlines(i, j, j + 1, colors='#FF0000', zorder=2, linewidth=0.25)

                            if i > len(y_part_linear) - 2:
                                pass
                            else:
                                if highlight[i + 1, j] == False:
                                    plt.hlines(i + 1, j, j + 1, colors='#FF0000', zorder=2, linewidth=0.25)

                part_area_part_threshold_15_colder = thermo_array_crop_part.copy()
                threshold_2_part_threshold_15_colder = set_temperature - threshold_limit - 5
                part_area_part_threshold_15_colder[
                    part_area_part_threshold_15_colder < threshold_2_part_threshold_15_colder] = 0
                ### drawing qwert
                part_area_part_threshold_15_colder[np.isnan(part_area_part_threshold_15_colder)] = avg
                highlight = (part_area_part_threshold_15_colder < 1)
                # highlight = np.ma.masked_less(highlight, 1)
                x_part_linear = np.linspace(0, size_of_part_x - 1, size_of_part_x)
                y_part_linear = np.linspace(0, size_of_part_y - 1, size_of_part_y)

                for i in range(len(y_part_linear)):
                    for j in range(len(x_part_linear)):
                        if highlight[i, j] == True:

                            if j == 0:
                                pass
                            else:
                                if highlight[
                                    i, j - 1] == False:  # the script goes through cell by cell. if there is a true value, it checks around the area if there is a false. if yes, a line is drawn
                                    plt.vlines(j, i, i + 1, colors='blue', zorder=2, linewidth=0.25)

                            if j > len(x_part_linear) - 2:
                                pass
                            else:
                                if highlight[i, j + 1] == False:
                                    plt.vlines(j + 1, i, i + 1, colors='blue', zorder=2, linewidth=0.25)

                            if i == 0:
                                pass
                            else:
                                if highlight[i - 1, j] == False:
                                    plt.hlines(i, j, j + 1, colors='blue', zorder=2, linewidth=0.25)

                            if i > len(y_part_linear) - 2:
                                pass
                            else:
                                if highlight[i + 1, j] == False:
                                    plt.hlines(i + 1, j, j + 1, colors='blue', zorder=2, linewidth=0.25)





                plt.grid('major')

                plt.colorbar()
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(8, 6)

                custom_lines = [Line2D([0], [0], color='b', lw=4),
                                Line2D([0], [0], color='#00D8FF', lw=4),
                                Line2D([0], [0], color='#FF7C00', lw=4),
                                Line2D([0], [0], color='#FF0000', lw=4)]
                very_cold = round(threshold_2_part_threshold_15_colder, 2)
                cold = round(threshold_2_part_threshold_10_colder, 2)
                hot = round(threshold_2_part_threshold_10_hotter, 2)
                very_hot = round(threshold_2_part_threshold_15_hotter, 2)
                plt.legend(custom_lines,
                           ["<" + str(very_cold) + "°C " + " ", "<" + str(cold) + "°C " + " ",
                            ">" + str(hot) + " " + "°C ",
                            ">" + str(very_hot) + "°C " + " "], bbox_to_anchor=(0, -0.23, 1, 0.2), loc="lower left",
                           mode="expand", ncol=4, prop={'size': 8})


                plt.savefig('plots/part_dynamic/' + str(filename) + "_part_dynamic" + ".png", bbox_inches='tight',
                            dpi=200)
                plt.savefig('plots/dashboard/part_dynamic/' + "demo_part_dynamic" + ".png", bbox_inches='tight', dpi=600)
                # plt.show()
                plt.clf()
                plot_time_6 = time.time()

                # show tresholded area
                plt.pcolormesh(part_area, cmap="viridis", vmin=0, vmax=max_T_part)
                plt.axis('scaled')
                plt.title("Plot 2D array, thresholded")

                x_axis(size_of_part_x)
                y_axis(size_of_part_y)

                if percentage_zero < percentage_for_ok_not_ok:
                    plt.text((size_of_part_x / 2), -20, "OK", fontsize=12)
                else:
                    plt.text((size_of_part_x / 2) - 6, -20, "not OK", fontsize=12)
                plt.colorbar()
                plt.grid('major')
                figure = plt.gcf()  # get current figure
                figure.set_size_inches(12, 8)
                plt.savefig('plots/threshold/' + str(filename) + "_threshold" + ".png", bbox_inches='tight', dpi=200)
                # plt.show()
                plt.clf()

                if ThreeD_graph == 1:

                    # 3D homogeneity graph this cannot be used, because the value of x and y is not the same, MAKES THE CODE VERY SLOW
                    x_3d = np.arange(0, h_tool_crop, 1)
                    y_3d = np.arange(0, w_tool_crop, 1)

                    xs, ys = np.meshgrid(x_3d, y_3d)
                    zs = thermo_tool_crop_np
                    zss = zs.transpose()
                    xs_size = len(xs[0]) - 1
                    ys_size = len(ys) - 1
                    layer = np.zeros((xs_size, ys_size))
                    layer[layer < 1] = avg

                    x_3d_layer = np.arange(0, ys_size, 1)
                    y_3d_layer = np.arange(0, xs_size, 1)
                    xs_layer, ys_layer = np.meshgrid(x_3d_layer, y_3d_layer)

                    fig = plt.figure()
                    ax = Axes3D(fig, auto_add_to_figure=False)
                    fig.add_axes(ax)

                    x_unmodified_ticks_part = list(range(0, h_tool_crop))
                    x_unmodified_ticks_part = np.array(x_unmodified_ticks_part)
                    x_modified_ticks_part = [item / mm_in_y for item in x_unmodified_ticks_part]
                    x_modified_ticks_part = [round(item, 1) for item in x_modified_ticks_part]

                    middle0 = x_modified_ticks_part[0]
                    middle1 = x_modified_ticks_part[math.floor((len(x_modified_ticks_part)) / 2)]
                    middle2 = x_modified_ticks_part[math.floor((len(x_modified_ticks_part) / 4))]
                    middle3 = x_modified_ticks_part[math.floor((len(x_modified_ticks_part) / 8))]
                    middle4 = x_modified_ticks_part[
                        math.floor((len(x_modified_ticks_part) / 4) + (len(x_modified_ticks_part) / 8))]
                    middle5 = x_modified_ticks_part[
                        math.floor((len(x_modified_ticks_part) / 2) + (len(x_modified_ticks_part) / 8))]
                    middle6 = x_modified_ticks_part[
                        math.floor((len(x_modified_ticks_part) / 2) + (len(x_modified_ticks_part) / 4))]
                    middle7 = x_modified_ticks_part[math.floor(
                        (len(x_modified_ticks_part) / 2) + (len(x_modified_ticks_part) / 4) + (
                                len(x_modified_ticks_part) / 8))]
                    middle8 = x_modified_ticks_part[-1]
                    x_tick = middle8
                    middle_list = []
                    middle_list.extend([middle0, middle2, middle1, middle6, middle8])

                    middleb0 = x_unmodified_ticks_part[0]
                    middleb1 = x_unmodified_ticks_part[math.floor((len(x_unmodified_ticks_part)) / 2)]
                    middleb2 = x_unmodified_ticks_part[math.floor((len(x_unmodified_ticks_part) / 4))]
                    middleb3 = x_unmodified_ticks_part[math.floor((len(x_unmodified_ticks_part) / 8))]
                    middleb4 = x_unmodified_ticks_part[
                        math.floor((len(x_unmodified_ticks_part) / 4) + (len(x_unmodified_ticks_part) / 8))]
                    middleb5 = x_unmodified_ticks_part[
                        math.floor((len(x_unmodified_ticks_part) / 2) + (len(x_unmodified_ticks_part) / 8))]
                    middleb6 = x_unmodified_ticks_part[
                        math.floor((len(x_unmodified_ticks_part) / 2) + (len(x_unmodified_ticks_part) / 4))]
                    middleb7 = x_unmodified_ticks_part[math.floor(
                        (len(x_unmodified_ticks_part) / 2) + (len(x_unmodified_ticks_part) / 4) + (
                                len(x_unmodified_ticks_part) / 8))]
                    middleb8 = x_unmodified_ticks_part[-1]
                    middle_list_b = []
                    middle_list_b.extend([middleb0, middleb2, middleb1, middleb6, middleb8])
                    plt.xticks((middle_list_b), (middle_list))
                    middle_list.clear()
                    middle_list_b.clear()

                    y_unmodified_ticks_part = list(range(0, w_tool_crop))
                    y_unmodified_ticks_part = np.array(y_unmodified_ticks_part)
                    y_modified_ticks_part = [item / mm_in_x for item in y_unmodified_ticks_part]
                    y_modified_ticks_part = [round(item, 1) for item in y_modified_ticks_part]

                    middle0 = y_modified_ticks_part[0]
                    middle1 = y_modified_ticks_part[math.floor((len(y_modified_ticks_part)) / 2)]
                    middle2 = y_modified_ticks_part[math.floor((len(y_modified_ticks_part) / 4))]
                    middle3 = y_modified_ticks_part[math.floor((len(y_modified_ticks_part) / 8))]
                    middle4 = y_modified_ticks_part[
                        math.floor((len(y_modified_ticks_part) / 4) + (len(y_modified_ticks_part) / 8))]
                    middle5 = y_modified_ticks_part[
                        math.floor((len(y_modified_ticks_part) / 2) + (len(y_modified_ticks_part) / 8))]
                    middle6 = y_modified_ticks_part[
                        math.floor((len(y_modified_ticks_part) / 2) + (len(y_modified_ticks_part) / 4))]
                    middle7 = y_modified_ticks_part[math.floor(
                        (len(y_modified_ticks_part) / 2) + (len(y_modified_ticks_part) / 4) + (
                                len(y_modified_ticks_part) / 8))]
                    middle8 = y_modified_ticks_part[-1]
                    y_tick = middle8
                    middle_list = []
                    middle_list.extend([middle0, middle2, middle1, middle6, middle8])

                    middleb0 = y_unmodified_ticks_part[0]
                    middleb1 = y_unmodified_ticks_part[math.floor((len(y_unmodified_ticks_part)) / 2)]
                    middleb2 = y_unmodified_ticks_part[math.floor((len(y_unmodified_ticks_part) / 4))]
                    middleb3 = y_unmodified_ticks_part[math.floor((len(y_unmodified_ticks_part) / 8))]
                    middleb4 = y_unmodified_ticks_part[
                        math.floor((len(y_unmodified_ticks_part) / 4) + (len(y_unmodified_ticks_part) / 8))]
                    middleb5 = y_unmodified_ticks_part[
                        math.floor((len(y_unmodified_ticks_part) / 2) + (len(y_unmodified_ticks_part) / 8))]
                    middleb6 = y_unmodified_ticks_part[
                        math.floor((len(y_unmodified_ticks_part) / 2) + (len(y_unmodified_ticks_part) / 4))]
                    middleb7 = y_unmodified_ticks_part[math.floor(
                        (len(y_unmodified_ticks_part) / 2) + (len(y_unmodified_ticks_part) / 4) + (
                                len(y_unmodified_ticks_part) / 8))]
                    middleb8 = y_unmodified_ticks_part[-1]
                    middle_list_b = []
                    middle_list_b.extend([middleb0, middleb2, middleb1, middleb6, middleb8])
                    plt.yticks((middle_list_b), (middle_list))
                    #   plt.gca().invert_yaxis()
                    middle_list.clear()
                    middle_list_b.clear()
                    #  ax.auto_scale_xyz([0, x_tick], [0, y_tick], [0, 400])
                    ax.plot_surface(xs, ys, zss, rstride=1, cstride=1, cmap='hot', zorder=2)
                    ax.plot_surface(ys_layer, xs_layer, layer, rstride=1, cstride=1, cmap='hot', alpha=0.2, zorder=1)
                    # plt.show()
                    ax.view_init(30, 45)
                    # disable auto rotation
                    ax.zaxis.set_rotate_label(False)
                    ax.set_zlabel('T [°C]', fontsize=10, rotation=0)
                    ax.yaxis.set_rotate_label(True)
                    ax.set_ylabel('y [mm]', fontsize=10, rotation=0)
                    ax.yaxis.labelpad = 20
                    ax.xaxis.set_rotate_label(True)
                    ax.set_xlabel('x [mm]', fontsize=10, rotation=0)
                    ax.xaxis.labelpad = 5
                    ax.set_box_aspect(
                        (np.ptp(xs), np.ptp(ys), np.ptp(zs)))  # aspect ratio is 1:1:1 in data space # works!
                    plt.savefig(str(filename) + "_hom_3D" + ".png", bbox_inches='tight')
                    plt.clf()
                else:
                    pass
                # os.system('plots.py') # to reduce the size of the main pyhton file, the plot script is stored here. This can be replaced by the actual script
                os.remove("excel/" + name)
                # Conversion script end
                # Conversion of phyton arrays to numpy arrays
                meanarray_np_tool = np.array(meanarray_tool)  # Converts meanarray to numpy array
                filenamearray_np_tool = np.array(filenamearray_tool)  # Converts namearray to numpy array
                homoarray_np_tool = np.array(homoarray_tool)
                skewarray_np_tool = np.array(skewarray_tool)
                st_dev_tool_np = np.array(st_dev_tool)
                kurtarray_np_tool = np.array(kurtarray_tool)
                meanerror_array_np_tool = np.array(meanerror_array_tool)
                homogeneity_line_array_np_tool = np.array(homogeneity_line_array_tool)
                w_tool_array_np = np.array(w_tool_array)
                h_tool_array_np = np.array(h_tool_array)
                homogeneity_percentage_tool_array_np = np.array(homogeneity_percentage_tool_array)

                meanarray_np_part = np.array(meanarray_part)  # Converts meanarray to numpy array
                filenamearray_np_part = np.array(filenamearray_part)  # Converts namearray to numpy array
                homoarray_np_part = np.array(homoarray_part)
                st_dev_part_np = np.array(st_dev_part)
                skewarray_np_part = np.array(skewarray_part)
                kurtarray_np_part = np.array(kurtarray_part)
                meanerror_array_np_part = np.array(meanerror_array_part)
                homogeneity_line_array_np_part = np.array(homogeneity_line_array_part)
                w_part_array_np = np.array(w_part_array)
                h_part_array_np = np.array(h_part_array)
                homogeneity_percentage_part_array_np = np.array(homogeneity_percentage_part_array)
                warning_zero_part_np = np.array(warning_zero_part)
                abc = np.stack((filenamearray_np_tool, meanarray_np_tool, homoarray_np_tool),
                               axis=-1)  # Stack column matrices together. Not used now.
                # Convesrion of numpy arrays to excel sheets
                e = pd.DataFrame(
                    {'File name': filenamearray_np_tool,
                     # Each of these lines create a new column with the array content
                     'Mean Values': meanarray_np_tool,
                     'Standard deviation': st_dev_tool_np,
                     '3D homogeneity': homoarray_np_tool,
                     'Homogeneity percentage': homogeneity_percentage_tool_array_np,
                     'Skewness': skewarray_np_tool,
                     'Kurtosis': kurtarray_np_tool,
                     'Mean homogeneity error?': meanerror_array_np_tool,
                     'Homogeneity line': homogeneity_line_array_np_tool,
                     'Size of cropped tool in x [mm]': w_tool_array_np,
                     'Size of cropped tool in y [mm]': h_tool_array_np,
                     })  # Black magic for pandas
                #   filename = filenamearray_part[0]
                filepath = f'results/{filename}.xlsx'  # Gives the filepath, TODO this wont work if multiple files are generated at once, and now it does work: solution was indentiation
                df = pd.read_csv("masterfile_demo.csv")
                df = pd.concat([e, df]).reset_index(drop=True)
                df.drop(df.tail(1).index, inplace=True)
                df.to_csv('masterfile_demo.csv', index=False)
                e.to_excel(filepath, index=False)  # converts DataFrame to excel
                e.to_csv('masterfile_instant_demo.csv', mode='a', index=False,
                         header=None)  # if needed: here write csv header



                # Convesrion of numpy arrays to excel sheets
                e_1 = pd.DataFrame(
                    {
                        'File name_part': filenamearray_np_part,
                        'Mean Values_part': meanarray_np_part,
                        'Standard deviation_part': st_dev_part_np,
                        '3D homogeneity_part': homoarray_np_part,
                        'Homogeneity percentage part': homogeneity_percentage_part_array_np,
                        'Skewness_part': skewarray_np_part,
                        'Kurtosis_part': kurtarray_np_part,
                        'Mean homog. error_part': meanerror_array_np_part,
                        'Homogeneity line_part': homogeneity_line_array_np_part,
                        'Size of cropped part in x [mm]': w_part_array_np,
                        'Size of cropped part in y [mm]': h_part_array_np,
                        'Warning system': warning_zero_part_np,
                    })  # Black magic for pandas
                #   filename = filenamearray_part[0]
                filepath = f'results/{filename}_part.xlsx'  # Gives the filepath, TODO this wont work if multiple files are generated at once, and now it does work: solution was indentiation
                df_1 = pd.read_csv("masterfile_demo_part.csv")
                df_1 = pd.concat([e_1, df_1]).reset_index(drop=True)
                df_1.drop(df_1.tail(1).index, inplace=True)
                df_1.to_csv('masterfile_demo_part.csv', index=False)
                e_1.to_excel(filepath, index=False)  # converts DataFrame to excel
                e_1.to_csv('masterfile_instant_part.csv', mode='a', index=False,
                           header=None)  # if needed: here write csv header
                ### important values
                percentage_for_ok = 100 - percentage_for_ok_not_ok
                threshold_limit_1 = threshold_limit + set_temperature
                threshold_limit_2 = threshold_limit + set_temperature + 5
                threshold_limit_3 = set_temperature - threshold_limit
                threshold_limit_4 = set_temperature - threshold_limit -5
                print('Mean temperature: ', mean_part)
                print('Standard deviation of the temperature: ', st_dev_part_1)
                print('Maximum temperature: ', max_T_part)
                print('Minimum temperature: ', min_T_part)
                print('Set temperature: ', set_temperature)
                print('Percentage for OK: ', percentage_for_ok)
                print('Threshold limit 1: ', threshold_limit_1)
                print('Threshold limit 2: ', threshold_limit_2)
                print('Threshold limit 3: ', threshold_limit_3)
                print('Threshold limit 4: ', threshold_limit_4)
                print('Name of file: ', filename)









            plt.close('all')
            plt.clf()


if __name__ == "__main__":
    print('Listening...')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path_folder = "./excel"
    event_handler = Event()
    observer = Observer()
    observer.schedule(event_handler, path_folder, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    # TODO: create a masterfile that only has the header in the beginning...but if it restarts, then a new file will be vreated. could be solved with an if it exists then dont create one. one more solution: just create the file beforehand