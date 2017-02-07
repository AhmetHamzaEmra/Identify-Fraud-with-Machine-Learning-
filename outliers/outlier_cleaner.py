
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    farklar=[]
    ### your code goes here
    for i in range(len(predictions)):
        fark[i]=abs(predictions[i]-net_worths[i])
    i=0
    while i> len(fark)//10:
        max=-1
        for j in range(len(fark)):
            if fark[j]>fark[max]:
                max=j
        
        predictions.remove(predictions[max])
        ages.remove(ages[max])
        net_worths.remove(net_worths[max])
        fark.remove(fark[max])
        
    
    cleaned_data.append()
    
    return cleaned_data
