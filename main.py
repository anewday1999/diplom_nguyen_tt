from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from lib.mainwindow import Ui_MainWindow
import pickle
import sys
from RandomForest import RandomForest
import pandas as pd

class mywindow(QMainWindow):
    #pyuic5 mainwindow.ui -o mainwindow.py
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.btn_check.clicked.connect(self.onCheckBtnClick)
        with open('./model/forest.model1', 'rb') as randomforestmodel:
            self.my_forest = pickle.load(randomforestmodel)
    
    def mapping(self, var, value):
        if var == 'Income_type':
            if value == 'Commercial associate':
                return 0
            if value == 'Pensioner':
                return 1
            if value == 'State servant':
                return 2
            if value == 'Student':
                return 3
            if value == 'Working':
                return 4
        if var == 'Education_type':
            if value == 'Academic degree':
                return 0
            if value == 'Higher education':
                return 1
            if value == 'Incomplete higher':
                return 2
            if value == 'Lower secondary':
                return 3
            if value == 'Secondary / secondary special':
                return 4
        if var == 'Family_status':
            if value == 'Civil marriage':
                return 0
            if value == 'Married':
                return 1
            if value == 'Separated':
                return 2
            if value == 'Single / not married':
                return 3
            if value == 'Widow':
                return 4
        if var == 'Housing_type':
            if value == 'Co-op apartment':
                return 0
            if value == 'House / apartment':
                return 1
            if value == 'Municipal apartment':
                return 2
            if value == 'Office apartment':
                return 3
            if value == 'Rented apartment':
                return 4
            if value == 'With parents':
                return 5
        if var == 'Occupation_type':
            if value == 'Accountants':
                return 0
            if value == 'Cleaning staff':
                return 1
            if value == 'Cooking staff':
                return 2
            if value == 'Core staff':
                return 3
            if value == 'Drivers':
                return 4
            if value == 'HR staff':
                return 5
            if value == 'High skill tech staff':
                return 6
            if value == 'IT staff':
                return 7
            if value == 'Laborers':
                return 8
            if value == 'Low-skill Laborers':
                return 9
            if value == 'Managers':
                return 10
            if value == 'Medicine staff':
                return 11
            if value == 'Other':
                return 12
            if value == 'Private service staff':
                return 13
            if value == 'Realty agents':
                return 14
            if value == 'Sales staff':
                return 15
            if value == 'Secretaries':
                return 16
            if value == 'Security staff':
                return 17
            if value == 'Waiters/barmen staff':
                return 18
        return None


    def onCheckBtnClick(self):
        try:
            Gender = int(self.ui.gender.currentText())
            Own_car = int(self.ui.car.currentText())
            Own_property = int(self.ui.property.currentText())
            Work_phone = int(self.ui.workphone.currentText())
            Phone = int(self.ui.phone.currentText())
            Email = int(self.ui.work_phone.currentText())

            Years_employed = float(self.ui.year_work.toPlainText())
            Num_children = int(self.ui.number_childrens.toPlainText())
            Num_family = int(self.ui.number_family.toPlainText())
            Account_length = int(self.ui.number_card.toPlainText())
            Total_income = float(self.ui.total_income.toPlainText())
            Age = float(self.ui.yearsold.toPlainText())

            Unemployed = int(self.ui.unemployee.currentText())
            Income_type = str(self.ui.type_income.currentText())
            Education_type = str(self.ui.type_education.currentText())
            Family_status = str(self.ui.famil_status.currentText())
            Housing_type = str(self.ui.house_type.currentText())
            Occupation_type = str(self.ui.work_type.currentText())
        except Exception as e:
            msgBox = QMessageBox()
            msgBox.setText(str(e))
            msgBox.show()
            msgBox.exec()
            return

        
        row = {'Gender' :Gender, 'Own_car' :Own_car, 'Own_property' :Own_property, 'Work_phone' :Work_phone, 'Phone' :Phone, 'Email' :Email, 'Unemployed' :Unemployed,
                'Num_children' :Num_children, 'Num_family' :Num_family, 'Account_length' :Account_length, 'Total_income' :Total_income, 'Age' :Age,
                'Years_employed' :Years_employed, 'Income_type' :self.mapping('Income_type', Income_type),
                'Education_type' :self.mapping('Education_type', Education_type), 'Family_status' :self.mapping('Family_status', Family_status),
                'Housing_type' :self.mapping('Housing_type', Housing_type), 'Occupation_type' :self.mapping('Occupation_type', Occupation_type)}
        try:
            df = pd.DataFrame(row, index=[0])
            result = self.my_forest.take_prediction(df.iloc[0])
            if result == 0:
                self.ui.lineEdit_result.setStyleSheet("background-color: rgb(155, 242, 145);")
                self.ui.lineEdit_result.setText('Низкий риск :)')
                
            else:
                self.ui.lineEdit_result.setStyleSheet("background-color: rgb(232, 95, 104);")
                self.ui.lineEdit_result.setText('Высокий риск :(')
            
        except Exception as e:
            msgBox = QMessageBox()
            msgBox.setText(str(e))
            msgBox.show()
            msgBox.exec()
            return

if __name__ == '__main__':
    app = QApplication([])
    application = mywindow()
    application.show()
    sys.exit(app.exec())