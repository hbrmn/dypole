# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 08:51:56 2021

@author: HB
"""

import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg

class MainWindow(qtw.QWidget):
    
    # making custom signals
    
    authenticated = qtc.pyqtSignal(str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # code goes here
        # label1 = qtw.QLabel('Input1')
        # label2 = qtw.QLabel('Input2')
        
        self.label1_input = qtw.QLineEdit()
        self.label2_input = qtw.QLineEdit()
        
        self.first_button = qtw.QPushButton('text')
        self.second_button = qtw.QPushButton('text2')
        # Layout for Widgets
        # layout = qtw.QHBoxLayout()
        # label1_layout = qtw.QHBoxLayout()
        # label1_layout.addWidget(label1)
        # label1_layout.addWidget(label1_input)
        
        # layout = qtw.QVBoxLayout()
        # layout.addLayout(label1_layout)
        # layout.addWidget(label2)

        # layout2 = qtw.QGridLayout()
        # layout2.addWidget(label2, 0, 0)
        
        layout3 = qtw.QFormLayout()
        layout3.addRow('FirstParameter', self.label1_input)
        layout3.addRow('SecondParameter', self.label2_input)
       
        button_widget = qtw.QWidget()
        button_widget.setLayout(qtw.QHBoxLayout())
        button_widget.layout().addWidget(self.first_button)
        button_widget.layout().addWidget(self.second_button)
        layout3.addRow('', button_widget)
        
        # button presses
        
        # Signal can be connected to everything in PyQt - function..etc.. doesnt have to be only callable like close()
        self.first_button.clicked.connect(self.test)
        self.second_button.clicked.connect(self.close)
        
        # using python function as QtSLot
        # @qtc.pyqtSlot(str)
        
        # Signals can carry data
        self.label1_input.textChanged.connect(self.set_button_text)
        
        # everything that happens during __init__ is not an instance method
        # i.e. self.something() will be gone after init is finished
        # this could be used to make a splashscreen
        
       # accessor method
        self.setLayout(layout3)
        
        # code ends here
        self.show()
        
    def set_button_text(self, text):
        if text:
            self.first_button.setText(f'Log In {text}')
        else:
            self.first_button.setText('Log In')
            
    def test(self):
        qtw.QMessageBox.information(self, 'Hello!', 'World')
    
    
if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    