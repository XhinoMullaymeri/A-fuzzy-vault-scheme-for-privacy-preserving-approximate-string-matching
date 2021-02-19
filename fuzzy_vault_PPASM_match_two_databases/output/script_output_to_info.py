

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
with open(files[0],"r") as f:
    file_lines = f.readlines()
    output_lines = output_lines + [",".join(line.split(":")) for line in file_lines[:-2]]  + [file_lines[-2][:-3] +","+ file_lines[-2][-3:]] + [file_lines[-1][:-3] +","+ file_lines[-1][-3:]]
    #print( file_lines[-2][:-3] +","+ file_lines[-2][-3:])

with open(files[2],"r") as f:
    file_lines = f.readlines()
    file_lines[0] =  file_lines[0].strip() + "\t, "+ hms_to_seconds(file_lines[0].split(":",1)[1].strip()) + " seconds"
    file_lines[1] =  file_lines[1].strip() + "\t, "+ hms_to_seconds(file_lines[1].split(":",1)[1].strip()) + " seconds"

    #print(len(file_lines))
    output_lines = output_lines +["\n"+50*"-"+"\n"+"Time info\n"+50*"-"]+ file_lines[:2]
    file_lines = [line for line in file_lines if line.strip()]

    #print(file_lines[999])    #file_lines[999] = TIMES FOR DECODING

######################################### EDITING ENCODING TIMES ################################################################################
    encoding_times_lines = file_lines[3:999]
    encoding_times_lines = [    re.search('\[(.+?)\]', line).group(1).split(",")       for line in encoding_times_lines]
    #each lines has 6 times
    #time for key to be created (key from secret and reference set)
    # #time for RS redudant
    #time for genuine projection --> fuzzy vault
    #time for adding chaff points --> fuzzy vault
    #time for shuffling --> fuzzy vault
    #total time

    total_encoding_times = len(encoding_times_lines)
    avg_time_secret_to_reference_set_key = sum([float(time[0]) for time in encoding_times_lines]) / total_encoding_times
    #print(avg_time_secret_to_reference_set_key)
    avg_time_redudant_symbols = sum([float(time[1]) for time in encoding_times_lines]) / total_encoding_times
    #print(avg_time_redudant_symbols)
    avg_time_creating_genuine_fuzzy_vault = sum([float(time[2]) for time in encoding_times_lines]) / total_encoding_times
    #print(avg_time_creating_genuine_fuzzy_vault)    
    avg_time_adding_chaff_points = sum([float(time[3]) for time in encoding_times_lines]) / total_encoding_times
    #print(avg_time_adding_chaff_points)
    avg_time_shuffling_fuzzy_vault = sum([float(time[4]) for time in encoding_times_lines]) / total_encoding_times
    #print(avg_time_shuffling_fuzzy_vault)
    avg_time_enc_total = sum([float(time[5]) for time in encoding_times_lines]) / total_encoding_times
    #print(avg_time_total)

    avg_encoding_times = [avg_time_secret_to_reference_set_key,avg_time_redudant_symbols,avg_time_creating_genuine_fuzzy_vault\
        ,avg_time_adding_chaff_points,avg_time_shuffling_fuzzy_vault,avg_time_enc_total]
    #print(avg_encoding_times)
    table_enc_details = [["Avg time secret to key:,",avg_encoding_times[0]],["Avg time creating redudant symbols:,",avg_encoding_times[1]],\
        ["Avg time projecting genuine points:,",avg_encoding_times[2]],["Avg time creating chaff points:,",avg_encoding_times[3]],\
            ["Avg time shuffling fuzzy vault:,",avg_encoding_times[4]],["Avg total time for encoding:,",avg_encoding_times[5]]]

######################################### EDITING ENCODING TIMES ################################################################################



######################################### EDITING DECODING TIMES ################################################################################

    decoding_time_lines = file_lines[1000:]
    total_decoding_times = len(decoding_time_lines) - 996
    decoding_time_lines = [ re.search('\[(.+?)\]', line).group(1).split(",") for line in decoding_time_lines if len(line.strip())>30]


    decodings_time_lines_not_keys = [time for time in decoding_time_lines if "not" in time[0]]  #only keys
    decodings_time_lines_keys = [time for time in decoding_time_lines if "is" in time[0]]       #only not keys

    sum_time_dec_key_total = [float(time[5]) for time in decodings_time_lines_keys]  #only total times of unlokcs
    avg_time_dec_key_total = sum(sum_time_dec_key_total) / len(sum_time_dec_key_total)  #avg total only for unlokcs
    #print("avg time for unlocks:",avg_time_dec_key_total)

    sum_time_dec_no_key_total = [float(time[5]) for time in decoding_time_lines if "not" in time[0] ]  #only total times of not keys
    avg_time_dec_no_key_total = sum(sum_time_dec_no_key_total) / len(sum_time_dec_no_key_total)  #avg total only for not keys
    #print("avg time for not keys :",avg_time_dec_no_key_total)    

    sum_time_dec_total = [float(time[5]) for time in decoding_time_lines ]  
    avg_time_dec_total = sum(sum_time_dec_total) / len(sum_time_dec_total)  #avg total only for all decodings
    #print("avg time for all decodings:",avg_time_dec_total)    

    avg_time_dec_secret_to_reference_key = sum([float(time[1]) for time in decoding_time_lines ]) / len(decoding_time_lines)
    avg_time_dec_rs_decoding = sum([float(time[2]) for time in decoding_time_lines ]) / len(decoding_time_lines)
    avg_time_dec_common_points = sum([float(time[3]) for time in decoding_time_lines ]) / len(decoding_time_lines)
    avg_time_dec_combination_interpolation_all = sum([float(time[4]) for time in decoding_time_lines ]) / len(decoding_time_lines)
    avg_time_dec_combination_interpolation_keys =  sum([float(time[4]) for time in decodings_time_lines_keys ]) / len(decodings_time_lines_keys)
    avg_time_dec_combination_interpolation_not_key =  sum([float(time[4]) for time in decodings_time_lines_not_keys ]) / len(decodings_time_lines_not_keys)

    #print(avg_time_dec_combination_interpolation_all)
    #print(avg_time_dec_combination_interpolation_not_key)
    #print(avg_time_dec_combination_interpolation_keys)
    #time for key to be created
    #time for RS decoding
    #time for common points
    #time for combinations
    #total time

    table_dec_details = [["Avg time secret to key:,",avg_time_dec_secret_to_reference_key],["Avg time for reedsolomon error correction:,",avg_time_dec_rs_decoding]\
        ,["Avg time for finding common points:,",avg_time_dec_common_points],["Avg time for combinations and polynomial reconstruction:,",avg_time_dec_combination_interpolation_all]\
            ,["Avg total decoding time:,",avg_time_dec_total],["Avg total decoding time(only for keys):,",avg_time_dec_key_total]\
             , ["Avg total decoding time(only for not keys):,",avg_time_dec_no_key_total]  \
            ,["Avg time for combinations and polynomial reconstruction (only for keys):,",avg_time_dec_combination_interpolation_keys]\
                ,["Avg time for combinations and polynomial reconstruction (only for not keys):,",avg_time_dec_combination_interpolation_not_key]]


# ######################################### EDITING DECODING TIMES ################################################################################



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
f = open("all_info_"+files[1].split("_",2)[2]+"_.txt", "w")
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

print("\n\t\tDetails about TP and common points on key",file=f)
TP_details_table = []
for key,value in dictionary_with_TP_common.items():
    temp = [key+",",str(value)+",",round((value/total_TP)*100,4)]
    TP_details_table.append(temp)

print(tabulate(TP_details_table,headers=["Common Points,","Number of TP cases,","Percentage of total TP cases (%)"]),file=f)


print("\n\t\tDetails about FP and common points on key",file=f)
FP_details_table = []
for key,value in dictionary_with_FP_common.items():
    temp = [key+",",str(value)+",",round((value/total_FP)*100,4)]
    FP_details_table.append(temp)

print(tabulate(FP_details_table,headers=["Common Points,","Number of FP cases,","Percentage of total FP cases (%)"]),file=f)




#PRINTING DETAILS ABOUT THE TIMES
print("",file=f)
print(tabulate(table_enc_details,headers=["Details for encoding,","\tTimes (s)\t"]),file=f)

print("",file=f)
print(tabulate(table_dec_details,headers=["Details for decoding,","\tTimes (s)\t"]),file=f)