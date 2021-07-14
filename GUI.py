#!/usr/bin/python3

from libra import *
import os, sys, pdb, time
from termcolor import colored
from Dialogs import Get_File, Get_Folder
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore, QtGui, QtWidgets


class ThreadsClass(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(int)

    def __init__(self, val):
        QtCore.QThread.__init__(self)
        self.val = val

    def __del__(self):
        self.wait()

    def run(self):
        self.progress_update.emit(self.val*100)


class update_progressbar_class(QtCore.QObject):
    progress_update = QtCore.pyqtSignal(int)

    def run(self, val):
        self.progress_update.emit(val*100)



def change_color_task_done(self, Color):
    palette = QtGui.QPalette()
    brush = QtGui.QBrush(Color)
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
    brush = QtGui.QBrush(Color)
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
    brush = QtGui.QBrush(Color)
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
    brush = QtGui.QBrush(Color)
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
    brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
    brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
    brush.setStyle(QtCore.Qt.SolidPattern)
    palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
    self.task_done.setPalette(palette)



class Ui_Form(object):
    def Run_Libra(self):
        self.threadclass.run(self.value_progressbar)
        self.task_done.setText("PLEASE WAIT")
        QApplication.processEvents()

        change_color_task_done(self, QtGui.QColor(150, 0, 0))

        if self.what_to_run.currentText() == "all":
            self.Total_Number_of_task = 8
        elif self.what_to_run.currentText() == "a_cnn+p_pre+p_cnn+b_pos+b_den":
            self.Total_Number_of_task = 7
        elif self.what_to_run.currentText() == "p_pre+p_cnn+b_pos+b_den":
            self.Total_Number_of_task = 5
        elif self.what_to_run.currentText() == "p_cnn+b_pos+b_den":
            self.Total_Number_of_task = 4
        elif self.what_to_run.currentText() == "b_pos+b_den":
            self.Total_Number_of_task = 3
        elif self.what_to_run.currentText() == "b_den":
            self.Total_Number_of_task = 2
        elif self.what_to_run.currentText() == "j_org":
            self.Total_Number_of_task = 2
        elif self.what_to_run.currentText() == "j_seg":
            self.Total_Number_of_task = 7
        self.current_taks_number = 0


        try:
            if self.printing.isChecked():
                printing = 0
            else:
                printing = 1

            if self.find_pacemaker.isChecked():
                find_pacemaker = 0
            else:
                find_pacemaker = 1

            if self.remove_intermediate_images.isChecked():
                remove_intermediate_images = "R" # remove
            else:
                remove_intermediate_images = "K" # keep

            if self.find_bottom.isChecked():
                find_bottom = 1 # find bottom
            else:
                find_bottom = 0

            Info = LIBRA()
            print(colored("[INFO] Starting LIBRA "+Info.version, 'green'))
            self.running_task_name.setText('LIBRA is running')


            Info.parse_args(["-i", self.path_to_input.text(),
                             "-o", self.path_to_output.text(),
                             "-m", self.path_to_Nets.text(),
                             "-sfnna", self.air_folder_name.text(),
                             "-sfnnp", self.pec_folder_name.text(),
                             "-sfntbm", self.breast_folder_name.text(),
                             "-sfnbd", self.density_folder_name.text(),
                             "-sfnfni", self.seg_name.text(),
                             "-not", self.n_of_threads.text(),
                             "-ng", self.n_gpu.text(),
                             "-mc", self.m_cpu.text(),
                             "-cm", self.core_multiplier.text(),
                             "-tbs", self.batch_size.text(),
                             "-po", str(printing),
                             "-fpm", str(find_pacemaker),
                             "-rii", remove_intermediate_images,
                             "-fb", str(find_bottom),
                             "-wttbd", self.what_to_run.currentText()])


            if Info.which_task_to_be_done == "all" or \
                Info.which_task_to_be_done=="a_cnn+p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="j_org" or \
                Info.which_task_to_be_done=="j_seg":
                self.running_task_name.setText('Getting Required Info')
                self.value_progressbar = self.current_taks_number/self.Total_Number_of_task
                self.threadclass.run(self.value_progressbar)
                QApplication.processEvents()
                Info.get_info_based_on_air_cnn()
                self.current_taks_number += 1


            if Info.which_task_to_be_done == "j_org":
                self.running_task_name.setText('Preprocessing just the original image')
                self.value_progressbar = self.current_taks_number/self.Total_Number_of_task
                self.threadclass.run(self.value_progressbar)
                QApplication.processEvents()
                Info.run_just_orginal_image_preprocessing()
                self.current_taks_number += 1


            if Info.which_task_to_be_done == "all" or \
                Info.which_task_to_be_done=="j_seg":
                self.running_task_name.setText('Air Segmentation Preprocessing')
                self.value_progressbar = self.current_taks_number/self.Total_Number_of_task
                self.threadclass.run(self.value_progressbar)
                QApplication.processEvents()
                Info.run_air_preprocessing()
                self.current_taks_number += 1


            if Info.which_task_to_be_done == "all" or \
                Info.which_task_to_be_done=="a_cnn+p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="j_seg":
                self.running_task_name.setText('Air Segmentation CNN')
                self.value_progressbar = self.current_taks_number/self.Total_Number_of_task
                self.threadclass.run(self.value_progressbar)
                QApplication.processEvents()
                Info.run_air_cnn()
                self.current_taks_number += 1


            if Info.which_task_to_be_done == "all" or \
                Info.which_task_to_be_done=="a_cnn+p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="b_pos+b_den" or \
                Info.which_task_to_be_done=="b_den" or \
                Info.which_task_to_be_done=="j_seg":
                self.running_task_name.setText('Getting Required Info')
                Info.model_path = Info.model_path_pec
                self.value_progressbar = self.current_taks_number/self.Total_Number_of_task
                self.threadclass.run(self.value_progressbar)
                QApplication.processEvents()
                Info.get_info_based_on_pec_cnn()
                self.current_taks_number += 1


            if Info.which_task_to_be_done == "all" or \
                Info.which_task_to_be_done=="a_cnn+p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="j_seg":
                self.running_task_name.setText('Pectoral Segmentation Preprocessing')
                self.value_progressbar = self.current_taks_number/self.Total_Number_of_task
                self.threadclass.run(self.value_progressbar)
                QApplication.processEvents()
                Info.run_pec_preprocessing()
                self.current_taks_number += 1


            if Info.which_task_to_be_done == "all" or \
                Info.which_task_to_be_done=="a_cnn+p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="j_seg":
                self.running_task_name.setText('Pectoral Segmentation CNN')
                self.value_progressbar = self.current_taks_number/self.Total_Number_of_task
                self.threadclass.run(self.value_progressbar)
                QApplication.processEvents()
                Info.run_pec_cnn()
                self.current_taks_number += 1


            if Info.which_task_to_be_done == "all" or \
                Info.which_task_to_be_done=="a_cnn+p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="b_pos+b_den" or \
                Info.which_task_to_be_done=="j_seg":
                self.running_task_name.setText('Breast Segmentation Postprocessing')
                self.value_progressbar = self.current_taks_number/self.Total_Number_of_task
                self.threadclass.run(self.value_progressbar)
                QApplication.processEvents()
                Info.run_breast_postprocessing()
                self.current_taks_number += 1


            if Info.which_task_to_be_done == "all" or \
                Info.which_task_to_be_done=="a_cnn+p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="p_pre+p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="p_cnn+b_pos+b_den" or \
                Info.which_task_to_be_done=="b_pos+b_den" or \
                Info.which_task_to_be_done=="b_den":
                self.running_task_name.setText('Breast Density Evaluation')
                self.value_progressbar = self.current_taks_number/self.Total_Number_of_task
                self.threadclass.run(self.value_progressbar)
                QApplication.processEvents()
                Info.run_feature_extraction()
                self.current_taks_number += 1


            self.value_progressbar = self.current_taks_number/self.Total_Number_of_task
            T_End = time.time()
            print("[INFO] The total elapsed time (for all files): "+'\033[1m'+ \
                  colored(str(round(T_End-Info.T_Start, 2)), 'red')+'\033[0m'+" seconds")
            print(colored("[INFO] *** The LIBRA Segmentation steps are performed SUCCESSFULY and the results are SAVED ***", 'green'))
            Comment = "The Requested Tasks Are Done"
            Comment1 = "LIBRA Done. No Task on queue"

        except:
            print(colored("LIBRA Failed!", "red"))
            Comment = "LIBRA Failed. Requested Tasks Are Done."
            Comment1 = "LIBRA Failed. No Task on queue."

            self.value_progressbar = self.current_taks_number/self.Total_Number_of_task

        self.task_done.setText(Comment)
        self.running_task_name.setText(Comment1)
        change_color_task_done(self, QtGui.QColor(0, 120, 0))
        self.threadclass.run(self.value_progressbar)
        QApplication.processEvents()



    def Update_ProgressBar(self):
        self.progressbar.setValue(self.value_progressbar*100)



    def Close_window(self):
        sys.exit(self)



    def press_in_path(self):
        self.ex = Get_Folder()
        self.ex.close()
        self.path_to_input.setText(self.ex.folderpath)

    def press_out_path(self):
        self.ex = Get_Folder()
        self.ex.close()
        self.path_to_output.setText(self.ex.folderpath)

    def press_Nets_path(self):
        self.ex = Get_Folder()
        self.ex.close()
        self.path_to_Nets.setText(self.ex.folderpath)

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(900, 600)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)

        self.frame = QtWidgets.QFrame(Form)
        self.frame.setGeometry(QtCore.QRect(10, 10, 511, 201))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        self.input_push = QtWidgets.QPushButton(self.frame)
        self.input_push.setGeometry(QtCore.QRect(410, 37, 91, 25))
        self.input_push.setObjectName("input_push")

        self.path_to_input = QtWidgets.QLineEdit(self.frame)
        self.path_to_input.setGeometry(QtCore.QRect(10, 40, 391, 25))
        self.path_to_input.setAlignment(QtCore.Qt.AlignCenter)
        self.path_to_input.setObjectName("path_to_input")

        self.output_push = QtWidgets.QPushButton(self.frame)
        self.output_push.setGeometry(QtCore.QRect(410, 97, 91, 25))
        self.output_push.setObjectName("output_push")

        self.path_to_output = QtWidgets.QLineEdit(self.frame)
        self.path_to_output.setGeometry(QtCore.QRect(10, 100, 391, 21))
        self.path_to_output.setAlignment(QtCore.Qt.AlignCenter)
        self.path_to_output.setObjectName("path_to_output")

        self.Nets_push = QtWidgets.QPushButton(self.frame)
        self.Nets_push.setGeometry(QtCore.QRect(410, 157, 91, 25))
        self.Nets_push.setObjectName("Nets_push")

        self.path_to_Nets = QtWidgets.QLineEdit(self.frame)
        self.path_to_Nets.setGeometry(QtCore.QRect(10, 160, 391, 21))
        self.path_to_Nets.setAlignment(QtCore.Qt.AlignCenter)
        self.path_to_Nets.setObjectName("path_to_Nets")

        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(10, 15, 391, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(10, 75, 391, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(15, 135, 391, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")

        self.label_13 = QtWidgets.QLabel(self.frame)
        self.label_13.setGeometry(QtCore.QRect(440, 7, 65, 16))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        self.label_13.setPalette(palette)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")

        self.frame_2 = QtWidgets.QFrame(Form)
        self.frame_2.setGeometry(QtCore.QRect(530, 220, 361, 321))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.frame_2.setFont(font)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")

        self.air_folder_name = QtWidgets.QLineEdit(self.frame_2)
        self.air_folder_name.setGeometry(QtCore.QRect(10, 40, 291, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.air_folder_name.setFont(font)
        self.air_folder_name.setAlignment(QtCore.Qt.AlignCenter)
        self.air_folder_name.setObjectName("air_folder_name")

        self.pec_folder_name = QtWidgets.QLineEdit(self.frame_2)
        self.pec_folder_name.setGeometry(QtCore.QRect(10, 100, 291, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.pec_folder_name.setFont(font)
        self.pec_folder_name.setAlignment(QtCore.Qt.AlignCenter)
        self.pec_folder_name.setObjectName("pec_folder_name")

        self.breast_folder_name = QtWidgets.QLineEdit(self.frame_2)
        self.breast_folder_name.setGeometry(QtCore.QRect(10, 160, 291, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.breast_folder_name.setFont(font)
        self.breast_folder_name.setAlignment(QtCore.Qt.AlignCenter)
        self.breast_folder_name.setObjectName("breast_folder_name")

        self.density_folder_name = QtWidgets.QLineEdit(self.frame_2)
        self.density_folder_name.setGeometry(QtCore.QRect(10, 220, 291, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.density_folder_name.setFont(font)
        self.density_folder_name.setAlignment(QtCore.Qt.AlignCenter)
        self.density_folder_name.setObjectName("density_folder_name")

        self.seg_name = QtWidgets.QLineEdit(self.frame_2)
        self.seg_name.setGeometry(QtCore.QRect(10, 280, 291, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.seg_name.setFont(font)
        self.seg_name.setAlignment(QtCore.Qt.AlignCenter)
        self.seg_name.setObjectName("seg_name")

        self.label_6 = QtWidgets.QLabel(self.frame_2)
        self.label_6.setGeometry(QtCore.QRect(10, 15, 291, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(self.frame_2)
        self.label_7.setGeometry(QtCore.QRect(10, 75, 291, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")

        self.label_8 = QtWidgets.QLabel(self.frame_2)
        self.label_8.setGeometry(QtCore.QRect(10, 135, 291, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")

        self.label_9 = QtWidgets.QLabel(self.frame_2)
        self.label_9.setGeometry(QtCore.QRect(10, 195, 291, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")

        self.label_109 = QtWidgets.QLabel(self.frame_2)
        self.label_109.setGeometry(QtCore.QRect(5, 255, 311, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_109.setFont(font)
        self.label_109.setAlignment(QtCore.Qt.AlignCenter)
        self.label_109.setObjectName("label_109")

        self.label_15 = QtWidgets.QLabel(self.frame_2)
        self.label_15.setGeometry(QtCore.QRect(290, 7, 71, 16))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        self.label_15.setPalette(palette)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")


        self.frame_3 = QtWidgets.QFrame(Form)
        self.frame_3.setGeometry(QtCore.QRect(10, 340, 511, 201))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")

        self.progressbar = QtWidgets.QProgressBar(self.frame_3)
        self.progressbar.setGeometry(QtCore.QRect(30, 120, 451, 31))
        self.progressbar.setMouseTracking(False)
        self.progressbar.setTabletTracking(False)
        self.progressbar.setAcceptDrops(False)
        self.progressbar.setAutoFillBackground(False)
        self.progressbar.setProperty("value", 0)
        self.progressbar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressbar.setTextVisible(True)
        self.progressbar.setObjectName("progressbar")

        self.label_17 = QtWidgets.QLabel(self.frame_3)
        self.label_17.setGeometry(QtCore.QRect(110, 90, 291, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")

        self.label_18 = QtWidgets.QLabel(self.frame_3)
        self.label_18.setGeometry(QtCore.QRect(440, 7, 61, 20))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        self.label_18.setPalette(palette)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_18.setFont(font)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")

        self.label_19 = QtWidgets.QLabel(self.frame_3)
        self.label_19.setGeometry(QtCore.QRect(110, 20, 291, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")

        self.running_task_name = QtWidgets.QLabel(self.frame_3)
        self.running_task_name.setGeometry(QtCore.QRect(110, 50, 291, 20))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 50))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        self.running_task_name.setPalette(palette)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.running_task_name.setFont(font)
        self.running_task_name.setAlignment(QtCore.Qt.AlignCenter)
        self.running_task_name.setObjectName("running_task_name")


        self.task_done = QtWidgets.QLabel(self.frame_3)
        self.task_done.setGeometry(QtCore.QRect(55, 160, 400, 20))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 120, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 120, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.task_done.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.task_done.setFont(font)
        self.task_done.setAlignment(QtCore.Qt.AlignCenter)
        self.task_done.setObjectName("task_done")

        self.run_push = QtWidgets.QPushButton(Form)
        self.run_push.setGeometry(QtCore.QRect(160, 550, 221, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.run_push.setFont(font)
        self.run_push.setObjectName("run_push")

        self.close_push = QtWidgets.QPushButton(Form)
        self.close_push.setGeometry(QtCore.QRect(600, 550, 221, 41))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(148, 148, 148))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        self.close_push.setPalette(palette)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.close_push.setFont(font)
        self.close_push.setObjectName("close_push")

        self.frame_4 = QtWidgets.QFrame(Form)
        self.frame_4.setGeometry(QtCore.QRect(530, 10, 361, 201))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")

        self.n_gpu = QtWidgets.QLineEdit(self.frame_4)
        self.n_gpu.setGeometry(QtCore.QRect(30, 40, 101, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.n_gpu.setFont(font)
        self.n_gpu.setAlignment(QtCore.Qt.AlignCenter)
        self.n_gpu.setObjectName("n_gpu")

        self.m_cpu = QtWidgets.QLineEdit(self.frame_4)
        self.m_cpu.setGeometry(QtCore.QRect(170, 40, 101, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.m_cpu.setFont(font)
        self.m_cpu.setAlignment(QtCore.Qt.AlignCenter)
        self.m_cpu.setObjectName("m_cpu")

        self.core_multiplier = QtWidgets.QLineEdit(self.frame_4)
        self.core_multiplier.setGeometry(QtCore.QRect(30, 100, 101, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.core_multiplier.setFont(font)
        self.core_multiplier.setAlignment(QtCore.Qt.AlignCenter)
        self.core_multiplier.setObjectName("core_multiplier")

        self.n_of_threads = QtWidgets.QLineEdit(self.frame_4)
        self.n_of_threads.setGeometry(QtCore.QRect(170, 100, 101, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.n_of_threads.setFont(font)
        self.n_of_threads.setAlignment(QtCore.Qt.AlignCenter)
        self.n_of_threads.setObjectName("n_of_threads")

        self.batch_size = QtWidgets.QLineEdit(self.frame_4)
        self.batch_size.setGeometry(QtCore.QRect(105, 160, 101, 21))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.batch_size.setFont(font)
        self.batch_size.setAlignment(QtCore.Qt.AlignCenter)
        self.batch_size.setObjectName("batch_size")

        self.label_10 = QtWidgets.QLabel(self.frame_4)
        self.label_10.setGeometry(QtCore.QRect(15, 15, 130, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")

        self.label_101 = QtWidgets.QLabel(self.frame_4)
        self.label_101.setGeometry(QtCore.QRect(165, 15, 100, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_101.setFont(font)
        self.label_101.setAlignment(QtCore.Qt.AlignCenter)
        self.label_101.setObjectName("label_101")

        self.label_11 = QtWidgets.QLabel(self.frame_4)
        self.label_11.setGeometry(QtCore.QRect(165, 75, 110, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")

        self.label_111 = QtWidgets.QLabel(self.frame_4)
        self.label_111.setGeometry(QtCore.QRect(28, 75, 100, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_111.setFont(font)
        self.label_111.setAlignment(QtCore.Qt.AlignCenter)
        self.label_111.setObjectName("label_111")

        self.label_12 = QtWidgets.QLabel(self.frame_4)
        self.label_12.setGeometry(QtCore.QRect(8, 135, 300, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")

        self.label_14 = QtWidgets.QLabel(self.frame_4)
        self.label_14.setGeometry(QtCore.QRect(275, 7, 80, 20))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        self.label_14.setPalette(palette)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")


        self.frame_5 = QtWidgets.QFrame(Form)
        self.frame_5.setGeometry(QtCore.QRect(10, 221, 511, 110))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")

        self.what_to_run = QtWidgets.QComboBox(self.frame_5)
        self.what_to_run.setGeometry(QtCore.QRect(10, 30, 211, 28))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.what_to_run.setFont(font)
        self.what_to_run.setObjectName("what_to_run")
        self.what_to_run.addItem("all")
        self.what_to_run.addItem("a_cnn+p_pre+p_cnn+b_pos+b_den")
        self.what_to_run.addItem("p_pre+p_cnn+b_pos+b_den")
        self.what_to_run.addItem("p_cnn+b_pos+b_den")
        self.what_to_run.addItem("b_pos+b_den")
        self.what_to_run.addItem("b_den")
        self.what_to_run.addItem("j_org")
        self.what_to_run.addItem("j_seg")

        self.label_16 = QtWidgets.QLabel(self.frame_5)
        self.label_16.setGeometry(QtCore.QRect(10, 10, 210, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_16.setFont(font)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")

        self.printing = QtWidgets.QCheckBox(self.frame_5)
        self.printing.setGeometry(QtCore.QRect(245, 33, 118, 20))
        self.printing.setObjectName("printing")

        self.find_pacemaker = QtWidgets.QCheckBox(self.frame_5)
        self.find_pacemaker.setGeometry(QtCore.QRect(360, 33, 118, 20))
        self.find_pacemaker.setObjectName("find_pacemaker")

        self.remove_intermediate_images = QtWidgets.QCheckBox(self.frame_5)
        self.remove_intermediate_images.setGeometry(QtCore.QRect(15, 70, 200, 20))
        self.remove_intermediate_images.setObjectName("remove_intermediate_images")

        self.find_bottom = QtWidgets.QCheckBox(self.frame_5)
        self.find_bottom.setGeometry(QtCore.QRect(245, 70, 250, 20))
        self.find_bottom.setObjectName("find_bottom")

        self.label_20 = QtWidgets.QLabel(self.frame_5)
        self.label_20.setGeometry(QtCore.QRect(366, 7, 136, 16))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        self.label_20.setPalette(palette)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_20.setFont(font)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")



        W_info = 15; H_info = 20
        self.info_input = QtWidgets.QPushButton(self.frame)
        self.info_input.setGeometry(QtCore.QRect(282, 15, W_info, H_info))
        self.info_input.setObjectName("info_input")
        self.info_input.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_input.setToolTip('This shows path to input dataset (images).')
        self.info_out = QtWidgets.QPushButton(self.frame)
        self.info_out.setGeometry(QtCore.QRect(280, 75, W_info, H_info))
        self.info_out.setObjectName("info_out")
        self.info_out.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_out.setToolTip('This shows the genral folder name for saving all results.')
        self.info_air_cnn = QtWidgets.QPushButton(self.frame)
        self.info_air_cnn.setGeometry(QtCore.QRect(302, 135, W_info, H_info))
        self.info_air_cnn.setObjectName("info_air_cnn")
        self.info_air_cnn.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_air_cnn.setToolTip('This is the previously trained networks path.' +
                                     '\n It should remain constant and pointing to the package path, '+
                                     '\n unless user retrains the network or moves the networks path.')


        self.info_GPU = QtWidgets.QPushButton(self.frame_4)
        self.info_GPU.setGeometry(QtCore.QRect(140, 15, W_info, H_info))
        self.info_GPU.setObjectName("info_GPU")
        self.info_GPU.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_GPU.setToolTip('This number shows number of GPUs being used in segmentation. '+
                                 '\n It is suggested to use one GPU. ' +
                                 '\n Also, 0 means running CNNs by CPU which is not suggested.')
        self.info_MCPU = QtWidgets.QPushButton(self.frame_4)
        self.info_MCPU.setGeometry(QtCore.QRect(260, 15, W_info, H_info))
        self.info_MCPU.setObjectName("info_MCPU")
        self.info_MCPU.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_MCPU.setToolTip('This number shows if user wants multi-CPU threaded processing or not. '+
                                 '\n 0 means running code on one CPU core '+
                                 '\n while 1 askes for maximum number of CPU cores available (this can make the'+
                                 'computer busy just by this task).')
        self.info_n_of_threads = QtWidgets.QPushButton(self.frame_4)
        self.info_n_of_threads.setGeometry(QtCore.QRect(270, 75, W_info, H_info))
        self.info_n_of_threads.setObjectName("info_n_of_threads")
        self.info_n_of_threads.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_n_of_threads.setToolTip('This number shows how many threads of CPU to be used for parrallel processing '+
                                 '\n on CPU for preprocessing. We suggest to select this number '+
                                 '\n a value between 5 to 15 depends on your computer free RAM.')
        self.info_multiplier_cpu = QtWidgets.QPushButton(self.frame_4)
        self.info_multiplier_cpu.setGeometry(QtCore.QRect(132, 75, W_info, H_info))
        self.info_multiplier_cpu.setObjectName("info_multiplier_cpu")
        self.info_multiplier_cpu.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_multiplier_cpu.setToolTip('This number shows how many times the parrallel set of images '+
                                            '\n (number of available cores * number of threads = one set) '+
                                            '\n should be processed before the Queue will be closed. '+
                                            '\n This number can vary based on maximum number of opened jobs. '+
                                            '\n it suggested to keep the number as follow: 1000/(2 * number of available cores * number of threads).'+
                                            '\n This number can be set between 3 to 6 for most of computers.')
        self.info_Batch = QtWidgets.QPushButton(self.frame_4)
        self.info_Batch.setGeometry(QtCore.QRect(302, 135, W_info, H_info))
        self.info_Batch.setObjectName("info_Batch")
        self.info_Batch.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_Batch.setToolTip('This is the number of images to be processed as a batch on GPU.')


        self.info_air_sav = QtWidgets.QPushButton(self.frame_2)
        self.info_air_sav.setGeometry(QtCore.QRect(277, 15, W_info, H_info))
        self.info_air_sav.setObjectName("info_air_sav")
        self.info_air_sav.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_air_sav.setToolTip('This is a subdirectory path for saving air segmentations resutls.')
        self.info_pec_sav = QtWidgets.QPushButton(self.frame_2)
        self.info_pec_sav.setGeometry(QtCore.QRect(298, 75, W_info, H_info))
        self.info_pec_sav.setObjectName("info_pec_sav")
        self.info_pec_sav.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_pec_sav.setToolTip('This is a subdirectory path for saving pectoral segmentations resutls.')
        self.info_breast_sav = QtWidgets.QPushButton(self.frame_2)
        self.info_breast_sav.setGeometry(QtCore.QRect(292, 135, W_info, H_info))
        self.info_breast_sav.setObjectName("info_breast_sav")
        self.info_breast_sav.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_breast_sav.setToolTip('This is a subdirectory path for saving breast segmentations resutls.')
        self.info_density_sav = QtWidgets.QPushButton(self.frame_2)
        self.info_density_sav.setGeometry(QtCore.QRect(297, 195, W_info, H_info))
        self.info_density_sav.setObjectName("info_density_sav")
        self.info_density_sav.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_density_sav.setToolTip('This is a subdirectory path for saving BIRADS and density map resutls.')
        self.info_seg_sav = QtWidgets.QPushButton(self.frame_2)
        self.info_seg_sav.setGeometry(QtCore.QRect(325, 255, W_info, H_info))
        self.info_seg_sav.setObjectName("info_seg_sav")
        self.info_seg_sav.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_seg_sav.setToolTip('This is a subdirectory path for saving the final segmentation summary.')


        self.info_print = QtWidgets.QPushButton(self.frame_5)
        self.info_print.setGeometry(QtCore.QRect(330, 33, W_info, H_info))
        self.info_print.setObjectName("info_print")
        self.info_print.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_print.setToolTip('This is path for printing results; '+
                                   '\n it is not suggested if you multi CPU processing.')
        self.info_find_pacemaker = QtWidgets.QPushButton(self.frame_5)
        self.info_find_pacemaker.setGeometry(QtCore.QRect(477, 33, W_info, H_info))
        self.info_find_pacemaker.setObjectName("info_find_pacemaker")
        self.info_find_pacemaker.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_find_pacemaker.setToolTip('If this is one, it will remove the pacemakers by replacing it with minimum.')
        self.info_task_sel = QtWidgets.QPushButton(self.frame_5)
        self.info_task_sel.setGeometry(QtCore.QRect(222, 30, W_info, H_info))
        self.info_task_sel.setObjectName("info_task_sel")
        self.info_task_sel.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:200px; min-width:10px; min-height:10px;''')
        self.info_task_sel.setToolTip('This shows which task/s to be performed; The options are: '+
                                      '\n all TO PROCESS all steps: a_air+a_cnn+p_pre+p_cnn+b_pos+b_den, '+
                                      '\n The other taks are as follow:'+
                                      '\n a_pre: preprocessing for background (air) segmentation, '+
                                      '\n a_cnn: CNN segmentation for background (air) segmentation, '+
                                      '\n p_pre: preprocessing for pectoral segmentation, '+
                                      '\n p_cnn: CNN segmentation for pectoral segmentation, '+
                                      '\n b_pos: postprocessing for fianl breast segmentation, '+
                                      '\n b_den: dense tissue segmentation, '+
                                      '\n j_seg: this does segmentation steps: a_air+a_cnn+p_pre+p_cnn+b_pos, '+
                                      '\n j_org: this recovers just the original image (as it can be removed by the check box too).')
        self.info_remove_intermediate_images = QtWidgets.QPushButton(self.frame_5)
        self.info_remove_intermediate_images.setGeometry(QtCore.QRect(212, 70, W_info, H_info))
        self.info_remove_intermediate_images.setObjectName("info_remove_intermediate_images")
        self.info_remove_intermediate_images.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_remove_intermediate_images.setToolTip('If checked the orginal images (which is huge) will be removed;'+
                                                        '\n otherwise, it will be remained in the image folder.'+
                                                        '\n This image size is almost like the DICOM file size.')
        self.info_find_bottom = QtWidgets.QPushButton(self.frame_5)
        self.info_find_bottom.setGeometry(QtCore.QRect(477, 70, W_info, H_info))
        self.info_find_bottom.setObjectName("info_find_bottom")
        self.info_find_bottom.setStyleSheet('''background-color: navy; color: white; border-color: navy;
                                        border-width:0.2px; border-radius:6px; font-size: 10px;
                                        border-style: solid; max-width:500px;
                                        max-height:100px; min-width:10px; min-height:10px;''')
        self.info_find_bottom.setToolTip('If this is checked, software tries to find the top portion of abdominal.')




        self.code_path = os.path.abspath(__file__)
        self.code_path,_ = os.path.split(self.code_path)
        self.retranslateUi(Form)



        ### connect things
        self.value_progressbar=0
        self.threadclass = update_progressbar_class()
        self.threadclass.progress_update.connect(self.progressbar.setValue)

        self.run_push.clicked.connect(self.Run_Libra)
        self.close_push.clicked.connect(self.Close_window)

        self.Nets_push.clicked.connect(self.press_Nets_path)
        self.input_push.clicked.connect(self.press_in_path)
        self.output_push.clicked.connect(self.press_out_path)

        QtCore.QMetaObject.connectSlotsByName(Form)




    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Deep LIBRA"))


        self.label_14.setText(_translate("Form", "Suggested"))
        self.label_10.setText(_translate("Form", "Number of GPUs"))
        self.label_101.setText(_translate("Form", "Multi CPUs"))
        self.label_111.setText(_translate("Form", "Core Multiplier"))
        self.label_11.setText(_translate("Form", "CPU Threads"))
        self.label_12.setText(_translate("Form", "Batch Size for Processing Images by CNNs"))
        self.info_GPU.setText(_translate("Form", "i"))
        self.info_MCPU.setText(_translate("Form", "i"))
        self.info_n_of_threads.setText(_translate("Form", "i"))
        self.info_multiplier_cpu.setText(_translate("Form", "i"))
        self.info_Batch.setText(_translate("Form", "i"))
        self.n_gpu.setText(_translate("Form", "0"))
        self.m_cpu.setText(_translate("Form", "0"))
        self.n_of_threads.setText(_translate("Form", "10"))
        self.core_multiplier.setText(_translate("Form", "5"))
        self.batch_size.setText(_translate("Form", "20"))


        self.label_15.setText(_translate("Form", "Optional"))
        self.label_6.setText(_translate("Form", "Folder Name for Saving Air Masks"))
        self.label_7.setText(_translate("Form", "Folder Name for Saving Pectoral Masks"))
        self.label_8.setText(_translate("Form", "Folder Name for Saving Breast Masks"))
        self.label_9.setText(_translate("Form", "Folder Name for Saving Breast Density"))
        self.label_109.setText(_translate("Form", "Folder Name for Saving Final Segmented Images"))
        self.info_air_sav.setText(_translate("Form", "i"))
        self.info_pec_sav.setText(_translate("Form", "i"))
        self.info_breast_sav.setText(_translate("Form", "i"))
        self.info_density_sav.setText(_translate("Form", "i"))
        self.info_seg_sav.setText(_translate("Form", "i"))
        self.air_folder_name.setText(_translate("Form", "air_net_data"))
        self.pec_folder_name.setText(_translate("Form", "pec_net_data"))
        self.breast_folder_name.setText(_translate("Form", "breast_temp_masks"))
        self.seg_name.setText(_translate("Form", "final_images"))
        self.density_folder_name.setText(_translate("Form", "breast_density"))


        self.label_20.setText(_translate("Form", "Performance Details"))
        self.label_16.setText(_translate("Form", "Select What to be performed"))
        self.info_print.setText(_translate("Form", "i"))
        self.info_task_sel.setText(_translate("Form", "i"))
        self.info_remove_intermediate_images.setText(_translate("Form", "i"))
        self.info_find_bottom.setText(_translate("Form", "i"))
        self.info_find_pacemaker.setText(_translate("Form", "i"))
        self.printing.setText(_translate("Form", "Print Flag"))
        self.find_pacemaker.setText(_translate("Form", "Remove Metals"))
        self.remove_intermediate_images.setText(_translate("Form", "Remove Intemediate Images"))
        self.find_bottom.setText(_translate("Form", "Remove Top Portion of Abdominal"))

        self.label_13.setText(_translate("Form", "Required"))
        self.info_input.setText(_translate("Form", "i"))
        self.info_out.setText(_translate("Form", "i"))
        self.info_air_cnn.setText(_translate("Form", "i"))
        self.input_push.setText(_translate("Form", "Open"))
        self.output_push.setText(_translate("Form", "Open"))
        self.Nets_push.setText(_translate("Form", "Open"))
        self.path_to_input.setText(_translate("Form", "/home/ohm/Desktop/Input"))
        self.path_to_output.setText(_translate("Form", "/home/ohm/Desktop/Output"))
        self.path_to_Nets.setText(_translate("Form", os.path.join(self.code_path,"Net")))
        self.label.setText(_translate("Form", "Path to Input Images"))
        self.label_2.setText(_translate("Form", "Path to Output Files"))
        self.label_3.setText(_translate("Form", "Path to Trained Networks"))


        self.label_18.setText(_translate("Form", "Monitor"))
        self.label_19.setText(_translate("Form", "Current Running Task"))
        self.label_17.setText(_translate("Form", "Progress for all sub_tasks"))
        self.running_task_name.setText(_translate("Form", "No Task on queue"))
        self.task_done.setText(_translate("Form", "The Requested Tasks Are Done"))


        self.run_push.setText(_translate("Form", "Run"))
        self.close_push.setText(_translate("Form", "Close"))
