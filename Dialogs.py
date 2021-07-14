from PyQt5.QtWidgets import QWidget, QFileDialog


class Get_File(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Direct to the file'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.openFileNameDialog()

        self.show()


    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Saved Model (*.h5)", options=options)
        self.close()


class Get_Folder(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Direct to the path'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.openFileNameDialog()
        self.show()


    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.folderpath = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.close()
