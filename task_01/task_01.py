#BMi Calc
try:
    a = int(input("Enter age: ")) 
    h = float(input("Enter height: "))  
    w = float(input("Enter weight: ")) 
    bmi = int(w/(h/100))

    print("Your BMI:", bmi)

    if bmi < 18.5:
        print("underweight")
    elif 18.5 <= bmi <= 24.9:
        print("normal weight")
    else:
        print("please consider a healthier lifestyle")

except ValueError:
    print("Value is not an int or float")
