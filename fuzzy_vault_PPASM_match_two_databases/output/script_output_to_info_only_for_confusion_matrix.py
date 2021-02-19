

from tabulate import tabulate
import os
import re
import csv

def hms_to_seconds(hms):
    #print(hms)
    hms = hms.split(":")

    return str(3600*int(hms[0]) + 60 * int(hms[1]) + int(hms[2]))



files = [filename for filename in os.listdir('.') if filename.startswith("output")]
#print(files)

output_lines = []


with open(files[2],"r") as f:
    file_lines = f.readlines()
    file_lines[0] =  file_lines[0].strip() + "\t, "+ hms_to_seconds(file_lines[0].split(":",1)[1].strip()) + " seconds"
    file_lines[1] =  file_lines[1].strip() + "\t, "+ hms_to_seconds(file_lines[1].split(":",1)[1].strip()) + " seconds"

total_TP = 0
total_TN = 0
total_FN = 0
total_FP = 0

dictionary_with_FP_common={"9":0,"10":0,"11":0,"12":0,"13":0,"14":0,"15":0,"16":0,"17":0,"18":0,"19":0,"20":0}
dictionary_with_TP_common={"9":0,"10":0,"11":0,"12":0,"13":0,"14":0,"15":0,"16":0,"17":0,"18":0,"19":0,"20":0}

with open(files[1],"r",encoding="utf-8") as f:
    file_lines_names = f.readlines()

    for line in file_lines_names:
        line = line.split('[',1)
        line_ncid = line[0].strip()
    
        matches = re.findall(r'\(.*?\)', line[1])

        if len(matches) == 0:
            total_FN = total_FN + 1
            total_TN = total_TN + 995   #995 should not match
            continue


        if len(matches) == 1:
            if line_ncid  in matches[0]:
                total_TP = total_TP + 1
                common_points = matches[0].split(",")[-1][:-1].strip()
                dictionary_with_TP_common[common_points] = dictionary_with_TP_common[common_points] + 1
                total_TN = total_TN + 995
            else:
                total_FP = total_FP + 1
                common_points = matches[0].split(",")[-1][:-1].strip()
                dictionary_with_FP_common[common_points] = dictionary_with_FP_common[common_points] + 1
                total_FN = total_FN + 1
                total_TN = total_TN + 994              

            continue

        is_TP = 0 # flag if we find a TP 
        for match in matches:
            
            if line_ncid  in match:
                is_TP = 1
                total_TP = total_TP + 1
                common_points = matches[0].split(",")[-1][:-1].strip()
                dictionary_with_TP_common[common_points] = dictionary_with_TP_common[common_points] + 1
            else:
                total_FP = total_FP + 1
                common_points = matches[0].split(",")[-1][:-1].strip()
                dictionary_with_FP_common[common_points] = dictionary_with_FP_common[common_points] + 1
            
        if is_TP == 0:
            total_TN = total_TN + 995 - len(matches) # + is_TP flag in case we've a TP
            total_FN = total_FN + 1
        else:
            total_TN = total_TN + 995 - len(matches) + 1 # 1 is tp



#confusion_matrix = [[total_TP,total_FN],[total_FP,total_TN]]
#print(tabulate(confusion_matrix,headers=["P,","\tN\t"]))

#PRINTING GENERAL DETAILS
f = open("_confusion_only"+files[1].split("_",2)[2]+"_.csv", "w")

print("test: "+files[1].split("_",2)[2],file=f)
for line in output_lines:
    print(line.strip(),file=f)

#PRINTING DETAILS ABOUT THE RECALL ETC
print("\n\t\tConfusion Matrix\n",file=f)
confusion_matrix = [["Actual Positive,",str(total_TP)+",",total_FN],["Actual Negative,",str(total_FP)+",",total_TN]]
print (tabulate(confusion_matrix,headers=['Actual/Pred,','Pred True,', 'Pred False']),file=f)

accuracy = (total_TP+total_TN) / (total_FN+total_FP+total_TP+total_TN)
threat_score = total_TP / (total_TP+total_FN+total_FP)
F1_score = (2*total_TP) / (2*total_TP + total_FP + total_FN)
recall = total_TP / (total_TP+total_FN)
precision = total_TP / (total_TP+total_FP) 

print("\n\tDerivations from confusion matrix",file=f)
derivation_table = [ ["Recall:,",recall] , ["Precision:,",precision] ,["Accuracy:,",accuracy] , ["Threat score:,",threat_score]  , \
    ["F1 score:,",F1_score]]
print (tabulate(derivation_table,headers=['index,','value,']),file=f)
