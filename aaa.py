def this_is(value, name="nul"):
    print("-" for _ in range(100))
    print(f"・{name} :")
    print(f"    type of {name} : {type(value)}")
    try:
       print(f"    shape of {name} : {value.shape}")
    except:
        print(f"    {name} has not shape.") 
    try:
        print(f"    length of {name} : {len(value)}")
    except:
        print(f"    {name} has not len()")
    print(f"    value : ")
    print(value)
    print()

def read_conditions(fname="train_conditions_1"):
    import joblib
    with open(fname, "rb") as f:     
        conditions = joblib.load(f)
    this_is(conditions, name="conditions")
    

if __name__ == "__main__":
    read_conditions()