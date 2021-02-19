#author Xhino Mullaymeri


import hashlib
import reedsolo
import random
from scipy.interpolate import lagrange
import numpy as np
from numpy import polyfit
from Levenshtein import distance
from operator import itemgetter
import statistics
import math
import itertools
import time
import string
import re
import matplotlib.pyplot as plt
import pickle
import csv
from tabulate import tabulate
import os



def secret_to_polynomial_coefficients(secret,**kwargs):
    """
    This function calculates polynomial's coefficients from secret. 
    Needs  secret. 
    If mode is given then depending on the mode some extra inputs might be need.
    If mode !=1 then polynomials desired degree has to be given too otherwise the degree is based on secret length.
    Params: string secret
    Params int mode , degree | default values mode = 1 and degree = 8
    Returns: list (int) coeffs
    """   

    coeffs=[]
    mode = kwargs.get('mode', 1)
    degree = kwargs.get('degree', 8)
    if mode == 1:
        coeffs = [ord(c) for c in secret]

    if mode != 1:
        secret = [str(ord(c)) for c in secret]


        while len(coeffs) <= degree:
            if len(secret) > mode:
                coeffs.append(''.join(secret[:mode]))
                secret = secret[mode:]
            else:
                coeffs.append( ''.join(secret))
                break

                
        while len(coeffs) <=degree:
            coeffs.append(1)

    coeffs = [int(coeff) for coeff in coeffs]
    return coeffs


def polynomial_coefficients_to_secret(coeffs,**kwargs):    
    """
    This function calculates secret from polynomial's coefficients. 
    Input:  polynomial coeffs. 
    If mode is given then depending on the mode some extra inputs might be need.
    If mode !=1 then polynomials  degree has to be given too otherwise the degree is considered to be the default one (8).
    Params: list coeffs
    Params int mode , degree | default values mode = 1 and degree = 8
    Returns: string secret
    """   
    secret=''
    mode = kwargs.get('mode', 1)
    degree = kwargs.get('degree', 8)

    for coeff in coeffs:

        if str(coeff) == '1':
            break

        secret = secret+str(coeff)

        
    secret = re.findall('..', secret)
    secret = ''.join([chr(int(num)) for num in secret])
    print(secret)
    return secret

    

def print_polynomial(coeffs):
    """
    This function has as input list of coeffs and prints a polynomial representation.
    Params: list(int) coeffs
    Retuns: none
    """
    s=''


    for index,value in enumerate(coeffs):

        power = str(len(coeffs)-(index+1))

        s += str(value) + 'x^' + power + ' + '
        
        # s += str(value)
        # break


    return s[:-2]


def evaluate_polynomial(x,coeffs,modulo):
    """
    This function takes an int x and evaluates polynomial for that x. Returns int y where y = P(x).
    Params: int x, list(int) coeffs , int modulo
    Returns: int y
    """   



    length = len(coeffs) - 1 
    coeffs = coeffs[::-1]
    y=0
    for power,coeff in enumerate(coeffs):
        y += coeff * pow(x,power)
        #y = y%modulo

    return y


def add_chaf_points(vault,number_of_chaff_points):
    """
    This function adds chaf points to the vault.
    Params:list(int) list_x, list(int) coeffs , int number_of_chaf, int max,list (tuples) vault
    Returns: none
    """   

    unzipped = list(zip(*vault))
    # centroid = (sum(list(unzipped[0])) / len(list(unzipped[0])), sum(list(unzipped[1])) / len(list(unzipped[0])))
    #calculating centroid
    positions = np.array(vault)
    centroid = tuple(positions.mean(axis = 0))


    #getting only Xs which are in range (-1,1)
    sample_x = []
    while len(sample_x) < number_of_chaff_points:
        point_x = np.random.normal(centroid[0], statistics.stdev(list(unzipped[0])))
        while point_x < -1 or point_x > 1:
            point_x = np.random.normal(centroid[0], statistics.stdev(list(unzipped[0])))
        sample_x.append(point_x)

    #checking if a fake X is same as a real one
    #if so he just remove it and remove equal number of fake Ys
    already_used_Xs = list(set(unzipped[0]) & set(sample_x))
    sample_x = list(set(sample_x).difference(already_used_Xs))
    sample_y = np.random.normal(centroid[1], statistics.stdev(list(unzipped[1])), number_of_chaff_points)
    sample_y = sample_y[len(already_used_Xs):]

    chaff_points = []
    for i in range(len(sample_x)):
        chaff_points.append((sample_x[i],sample_y[i]))

    vault = vault + chaff_points

    # if input('Show fuzzy vault plot?')=='y':
    #     fig, ax = plt.subplots()
    #     ax.set_title('{!r}'.format('Fuzzy Vault'), color='C0')
    #     ax.plot(sample_x, sample_y,'.',label='CHAFF POINTS',color ='red',ms = 15 )
    #     ax.plot(list(unzipped[0]),list(unzipped[1]),'.',label='GENUINE POINTS',color='green',ms = 15)
    #     ax.legend()
    #     fig.show()

    return vault

def add_chaf_points_using_reference_set(vault,number_of_chaff_points,reference_set,ref_chaff_perc,initial_polynomial):
    """
    This function adds chaf points to the vault.
    Params:list(int) list_x, list(int) coeffs , int number_of_chaf, int max,list (tuples) vault
    Returns: none
    """   

    unzipped = list(zip(*vault))
    # centroid = (sum(list(unzipped[0])) / len(list(unzipped[0])), sum(list(unzipped[1])) / len(list(unzipped[0])))
    #calculating centroid
    positions = np.array(vault)
    centroid = tuple(positions.mean(axis = 0))


    number_of_ref_chaff_points = int(number_of_chaff_points * ref_chaff_perc)
    number_of_random_chaff_points = number_of_chaff_points - number_of_ref_chaff_points

    sample_x = []
    while len(sample_x) < number_of_random_chaff_points:
        point_x = np.random.normal(centroid[0], statistics.stdev(list(unzipped[0])))
        while point_x < -1 or point_x > 1:
            point_x = np.random.normal(centroid[0], statistics.stdev(list(unzipped[0])))
        sample_x.append(point_x)

    #checking if a fake X is same as a real one
    #if so he just remove it and remove equal number of fake Ys
    already_used_Xs = list(set(unzipped[0]) & set(sample_x))
    sample_x =  list(set(sample_x).difference(already_used_Xs))

    total_available_x_chaffs = []

    #dict_doubles={}
    # for key,value in dict_doubles.items():
    #     if len(value) > 1:
    #         print(key,value)

    for word in reference_set:
        x = dummy_word_hash(word)
        random.seed(x)
        x = random.randint(0,1000000)

        # if x not in dict_doubles.keys():
        #     dict_doubles[x]=[word]
        # else:
        #     dict_doubles[x].append(word)


        total_available_x_chaffs.append(x)

    total_available_x_chaffs =[math.cos(x) for x in total_available_x_chaffs]
    total_available_x_chaffs = list(set(total_available_x_chaffs))   #set
    random.seed(a = time.time())
    random_x_chaffs = set(random.sample(total_available_x_chaffs,k = number_of_ref_chaff_points ))

    while len(set(unzipped[0]).intersection(random_x_chaffs)) !=0 or len(random_x_chaffs) != number_of_ref_chaff_points:
        random.seed(a = time.time())
        random_x_chaffs = set(random.sample(total_available_x_chaffs,k = number_of_ref_chaff_points))

    sample_x = sample_x + list(random_x_chaffs)

    #checking if a fake X is same as a real one
    #if so he just remove it and remove equal number of fake Ys8

    sample_y = np.random.normal(centroid[1], statistics.stdev(list(unzipped[1])), number_of_chaff_points)
    #sample_y = [ random.choice([-1,1])*(round(random.random(),5)) * max(unzipped[1])*random.randint(-5,5)   for i in range(number_of_chaff_points)]
    sample_y = sample_y[:len(sample_x)]

    chaff_points = []
    for i in range(len(sample_x)):
        chaff_points.append((sample_x[i],sample_y[i]))

    vault = vault + chaff_points

    if input('Show fuzzy vault plot?')=='y':
        fig, ax = plt.subplots()
        ax.set_title('{!r}'.format('Fuzzy Vault'), color='C0')
        ax.plot(sample_x, sample_y,'.',label='CHAFF POINTS',color ='red',ms = 15 )
        ax.plot(list(unzipped[0]),list(unzipped[1]),'.',label='GENUINE POINTS',color='green',ms = 15)
        ax.legend()


        # x = np.linspace(-1, 1, 50)
        # y = [initial_polynomial(i) for i in x]
        # ax.plot(x, y)



        fig.show()




    return vault


def create_lock_key(n):
    
    #All uppercase English letters in a lsit
    alphabet = list(string.ascii_uppercase)

    #creating a random secret
    key = ''
    for _ in range(n):
        key += random.choice(alphabet) 
    
    return key


def str_to_ngarms(string,n):
    '''
    This function gets a str and an int n and returns a list
    with ngrams
    :param string: str that we want to create ngrams from
    :param n: length of ngrams
    :return: list with ngrams
    '''
    string = ' '*(n-1) + string +' '*(n-1)

    ngram_list = [string[i:i + n] for i in range(len(string) - (n-1))]

    # for i in range(len(string) - (n-1)):
    #     ngram_list.append(string[i:i + n])

    return ngram_list

def create_reference_set_from_file(path):
    ''' 
    Creates and returns a list-reference set from a given file. Basically preprocess the file 
    (strips,Cases,etc)
    :param path: File's path from which reference set will be created
    :return: list "reference set" 
    '''
    with open(path) as file: 
        lines = [line.strip().upper() for line in file.readlines()]
    return lines

def calculate_edit_distance_levenshtein(secret,reference_set):
    '''
    Calculates and returns a list which contains levenshtein distances of a reference set of strings  with another string.
    The list is returned sorted (ascending)
    :param secret: str which will be compared with reference set
    :param reference_set: list of a reference set
    '''

    reference_set_distances=[]
    for item in reference_set:
        reference_set_distances.append((item,distance(secret,item)))
    
    reference_set_distances = sorted(reference_set_distances, key=lambda tup: tup[1])
    #print(reference_set_distances)
    return reference_set_distances

def calculate_similarity_jaccard(secret,ngram,reference_set):
    '''
    Calculates and returns a list which contains jaccard similarity of a reference set of strings  with another string.
    The list is returned sorted (ascending)
    :param secret: str which will be compared with reference set
    :param ngram: size of ngrams that will be created in order to find similarity
    :param reference_set: list of a reference set
    '''
    reference_set_similarity=[]

    for item in reference_set:
        #print(str_to_ngarms(secret,ngram))
        intersection_card = len(set.intersection(*[set(str_to_ngarms(secret,ngram) ),set(str_to_ngarms(item,ngram))])) 
        union_card = len(set.union(*[set(str_to_ngarms(secret,ngram)),set(str_to_ngarms(item,ngram))])) 
        reference_set_similarity.append((item,intersection_card/float(union_card)))
    
    reference_set_similarity = sorted(reference_set_similarity, reverse = True, key=lambda tup: tup[1])
    #print(reference_set_similarity)   
    return reference_set_similarity



def dummy_word_hash(word):

    hashed_word_number = []
    for index,value in enumerate(word):
        #hashed_word_number.append(ord(value)*(index+1)*2^(index+1))
        hashed_word_number.append((ord(value)*(index+1))^(index+8))
        #hashed_word_number.append((ord(value)))
        #hashed_word_number.append(abs(hash(value)%(10 ** 8)))

    return sum(hashed_word_number)

def secret_to_key(secret,key_length,reference_set):

    if similarity_type[-1]=='m':
        distances = calculate_similarity_jaccard(secret,1,reference_set)
    elif similarity_type[-1]=='s':
        distances = calculate_similarity_jaccard(secret,2,reference_set)
    else:
        distances = calculate_edit_distance_levenshtein(secret,reference_set)
    #distances = calculate_edit_distance_levenshtein(secret,reference_set)
    
    dict_reference_similarity={}
    for element in distances[:key_length]:
        dict_reference_similarity[element[0]]= round(element[1],4)

    #for key,value in temp.items():
    #    print(key,':',value)

    hashes = {}
    for tupple in distances:
        hashes[tupple[0]] = dummy_word_hash(tupple[0])

    hashes = [ dummy_word_hash(tupple[0]) for tupple in distances]
    return hashes[:key_length],dict_reference_similarity

def alice(reference_set):
    
    #creating the secret
    #secret = create_secret(k = 4)   #random secret
    
    secret = input('\nsecret: ').upper()
    txt ='The secret string: "{}"'
    print(txt.format(secret))
    print("\n")

    #if input('Secret created...') == 'y':
    #    print(secret)

    #creating the key
    key ,dict_reference_similarity_user_A= secret_to_key(secret,key_length,reference_set)
    #print(key)
    
    #rsc codec 
    reed_solomon_redudant_coeffs=[]
    if ecc_symbols > 0:
        rsc = reedsolo.RSCodec(ecc_symbols)
        #encode secret UTF-8 first we make string to bytes
        enc_secret = rsc.encode(secret.encode())
        reed_solomon_redudant_coeffs=[item for item in enc_secret][-ecc_symbols:]
        msg = "Secret encoded in rs("+str(ecc_symbols)+"+k,k)..."
        
        input()("Redundant Data:",reed_solomon_redudant_coeffs)
    
    # if input(msg) =='y':
    #     print(enc_secret)
    #     print([item for item in enc_secret])


    #Creating polynomial
    #getting coeffs from secret (ascii numbers) 
    polynomial_coefficients = secret_to_polynomial_coefficients(secret,mode=3,degree=polynomial_degree)
    concatenated_coeffs = "".join(str(coeff) for coeff in polynomial_coefficients)
    polynomial_coefficients_hash = hashlib.md5(concatenated_coeffs.encode()).hexdigest() 

    #print("\nPolynomial Representation of the string\n")
    poly_string_repr = print_polynomial(polynomial_coefficients)
    #print(print_polynomial(polynomial_coefficients))

    #print("\nPolynomial Validation Hash\n")
    #print(polynomial_coefficients_hash)

    input("\nPolynomial info\n")
    print(tabulate([["Polynomial Representation of the string:",poly_string_repr]\
        ,["Polynomial Validation Hash:",polynomial_coefficients_hash]],tablefmt="pretty"))


    initial_polynomial = np.poly1d(polynomial_coefficients)
    # if input('Coefficients calculated...')=='y':
    #     print_polynomial(polynomial_coefficients)
    #     #print(initial_poly)

    
    #List of Xs
    list_Xs = key
    #print(list_Xs)
    temp=[]
    for x in list_Xs:
        random.seed(a=x)
        temp.append(random.randint(0,1000000))
    list_Xs = temp
    #print(list_Xs)

    #if input('Key mapping --> list_Xs...')=='y':
    list_Xs =[math.cos(x) for x in list_Xs]
    list_Xs = [round(x,round_decimal_x) for x in list_Xs]

    table_key_A = []
    for index,value in enumerate (dict_reference_similarity_user_A.items()):
        temp = [index,value[0],value[1],list_Xs[index]]
        table_key_A.append(temp)

    #list_Xs = list(set(list_Xs))

    #print('\n\n')
    #print("Polynomial Representation of the string:",poly_string_repr)
    input("\nkey info\n")
    print(tabulate(table_key_A,headers=["Reference index","Reference String","Similarity Level","Cosine value"],tablefmt="pretty"))
    print('\n\n')
    #print("Polynomial Validation Hash:",polynomial_coefficients_hash)
    #print('\n\n')


        #print("std dev: ",statistics.stdev(list_Xs))
        #print('mapped: ',list_Xs)


    #List of Ys
    list_Ys = [] 
    for i in list_Xs:
        y = round(initial_polynomial(i),round_decimal_y)
        list_Ys.append(y)

    #if input('list_Ys calculated. print?')=='y':
    #    print(list_Ys)

    #Creating Vault
    vault =[]
    for i in range(len(list_Xs)):
        vault.append((list_Xs[i],list_Ys[i]))
    #if input('Vault created. (x,y) tupples...')=='y':
    #    print(vault)


    if chaf_point_generator == 0:
        add_chaf_points(vault,number_of_chaff_points)

    if chaf_point_generator == 1:
        vault = add_chaf_points_using_reference_set(vault,number_of_chaff_points,reference_set,ref_chaff_perc,initial_polynomial)
    

    #if input('Vault added chaf points...')=='y':
    #    print(vault)

    random.shuffle(vault)

    t_shuffled  =  time.time()
    #if input('Vault shuffled...')=='y':
    #    print(vault)

    return vault,list_Xs,list_Ys,initial_polynomial,reed_solomon_redudant_coeffs,polynomial_coefficients,secret,polynomial_coefficients_hash


def bob_first_phase(reed_solomon_redudant_coeffs):

    bob_key = input("Enter User's B string :")
    bob_key = bob_key.upper()
    temp1 = bob_key

    if input("Use reed solomon error correction data?:") =='y':
        try:
            rsc = reedsolo.RSCodec(ecc_symbols)
            codeword = [coeff for coeff in bob_key.encode()] + reed_solomon_redudant_coeffs
            print("codeword:",codeword)
            dec_seret = rsc.decode(codeword)
            bob_key = dec_seret[0].decode("utf-8")
        except:
            print("\nCould not do reed-solomon decoding\n")

    Bob_list_x ,dict_reference_similarity_user_A = secret_to_key(bob_key,key_length,reference_set)
    
    print("\n\n\n")
    txt ='user_B string: "{}"'
    print(txt.format(bob_key))
    print("\n\n\n")

    # if input("Use reed solomon error correction decoder?:") =='y':
    #     rsc = reedsolo.RSCodec(ecc_symbols)
    #     codeword = [coeff for coeff in bob_key.encode()] + reed_solomon_redudant_coeffs
    #     print(codeword)
    #     dec_seret = rsc.decode(codeword)
    #     bob_key = dec_seret[0].decode("utf-8") 
    #     print(bob_key)

    temp=[]
    for x in Bob_list_x:
        random.seed(a=x)
        temp.append(random.randint(0,1000000))
    Bob_list_x = temp
    Bob_list_x =[math.cos(x) for x in Bob_list_x]
    Bob_list_x = [round(x,round_decimal_x) for x in Bob_list_x]
    #Bob_list_x = list(set(Bob_list_x))
    #print("std dev: ",statistics.stdev(Bob_list_x))


    table_key_B = []
    for index,value in enumerate (dict_reference_similarity_user_A.items()):
        temp = [index,value[0],value[1],Bob_list_x[index]]
        table_key_B.append(temp)

    print(tabulate(table_key_B,headers=["Reference index","Reference String","Similarity Level","Cosine value"],tablefmt="pretty"))



    # try:
    #     rsc = reedsolo.RSCodec(ecc_symbols)
    #     codeword = [coeff for coeff in bob_key.encode()] + reed_solomon_redudant_coeffs
    #     dec_seret = rsc.decode(codeword)
    #     bob_key = dec_seret[0].decode("utf-8")

    #     # if bob_key != temp1:
    #     #     number = number +1 
    #     #     print(temp1, bob_key)
    #     #     #print(len(temp1),len(bob_key))
    #     #     #print([c for c in temp1])
    #     #     #print([c for c in bob_key])
    #     # else:
    #     #     print(bob_key,temp1)

    # except:
    #     pass
    # finally:
    #     t_rs_decoding = time.time()

    return  Bob_list_x


def matching(vault,Bob_list_x):

    unzipped_vault = list(zip(*vault))
    vault_x = unzipped_vault[0]
    vault_y = unzipped_vault[1]

    common_points=[]

    for index, value in enumerate (vault_x):
        if value in Bob_list_x:
            common_points.append((value,vault_y[index]))
    

    return common_points


def bob_second_phase(polynomial_coefficients_hash,common_points,rsc,polynomial_degree,polynomial_coefficients):
    global names_that_matched

    combinations = list(itertools.combinations(common_points,polynomial_degree+1))
    
    #print("Common points:",len(common_points),"Total Combinations:",len(combinations))

    number_of_corrects = 0 
    for index,comb in enumerate(combinations):
        #print("comb_",index)
        unzipped = list(zip(*comb))

        reconstructed_polynomial = lagrange(list(unzipped[0]), list(unzipped[1]))
        
        #reconstructed_polynomial = np.polyfit(list(unzipped[0]), list(unzipped[1]), 8)
        reconstructed_polynomial_coefficients=[]
        
        for coeff in reconstructed_polynomial.coefficients:
            coeff = round(coeff)
            coeff = int(coeff)
            reconstructed_polynomial_coefficients.append(coeff)

        first_usefull_coeff = next((index for index,value in enumerate(reconstructed_polynomial_coefficients) if value != 0), -1)
        usefull_coeffs = reconstructed_polynomial_coefficients[-(len(reconstructed_polynomial_coefficients)-first_usefull_coeff):]
        diff = [abs(x1 - x2) for (x1, x2) in zip(usefull_coeffs, polynomial_coefficients)]


        concatenated_coeffs = "".join(str(coeff) for coeff in usefull_coeffs)
        interpolating_polynomial_coefficients_hash = hashlib.md5(concatenated_coeffs.encode()).hexdigest() 

        if polynomial_coefficients_hash == interpolating_polynomial_coefficients_hash:
            print("interpolating polynomial coeffs:",usefull_coeffs)
            print(polynomial_coefficients_hash,"==",interpolating_polynomial_coefficients_hash)
            print("Number of common points:",len(common_points))
            print("match")
            return
            
    print("Number of common points:",len(common_points))
    print("not match")


#GLOBALS   
number_of_chaff_points = 150
ref_chaff_perc = 1   #percentage of chaff points which are direct Xs from the reference set e.g. if 0.3 then 70% of chaffs will be totally random
ecc_symbols = 4
polynomial_degree = 8
key_length = 15
round_decimal_x = 6
round_decimal_y = 8
names_that_matched=[]
dictionary_with_names_and_matched_names = {}
similarity_type =""
chaf_point_generator = 1
interpolation_method = "Lagrange interpolation"

def test_info_input():

    #reference set 
    # index 0 has reference set number and #index 0 has its type
    reference_set_params = [2,25]
    #reference_set_params  = input("Reference set params... e.g.  1,20: ").split(',')
    #reference_set_name = 'reference_set'+str(reference_set_params[0])+'_'+str(reference_set_params[1])+'x26.txt'
    #reference_set_name = input("enter reference set name (with .txt): ")
    reference_set_name = "reference_set.txt"
    #selecting similarity algorithm
    similarity_type = int(input("Similarity algorithm type 1 , 2 or 3 (jaccard_1gram,jaccard_2grams,levensthein): "))
    similarity_type = 1    
    if similarity_type == 1:
        similarity_type = 'jaccard_1gram'
    elif similarity_type == 2:
        similarity_type = 'jaccard_2grams'
    elif similarity_type == 3:
        similarity_type = 'levenshtein'
    else:
        print("default selection : jaccard_1gram")
        similarity_type = 'jaccard_1gram'

    #key size
    key_length = int(input("enter key length:"))

    #data_set_encoding = input("Enter encoding data set's name: ")
    #data_set_decoding = input("Enter decoding data set's name: ")
    data_set_encoding = "none"
    data_set_decoding = "none"

    polynomial_degree = int(input("Enter polynomial degree: "))
    #interpolation_method = input("Choose lagrange or polyfit for polynomial reconstruction: ")
    number_of_chaff_points = int(input("Enter number of chaff points: "))
    ecc_symbols = int(input("Enter reed solomon redudant symbols number: "))
    chaf_point_generator = int(input("select chaff point generator (0-->normal noise , 1--> reference set based abscossas): "))
    #round_decimal_x = int(input("Enter round deciman of abscissa: "))
    #round_decimal_y = int(input("Enter round deciman of ordinate: "))

    print("\n\n------------------------Test info------------------------")
    table = [["Data set encoding:",data_set_encoding],["Data set decoding:",data_set_decoding],\
        ["Reference set:",reference_set_name],["Similarity algorithm:",similarity_type] ,["Polynomial degree:",polynomial_degree],["Polynomial reconstruction:",interpolation_method],\
            ["Key size:",key_length],["Chaff points:",number_of_chaff_points],["Reed solomon redudant symbols:",ecc_symbols],["Round decimal of abscissa",round_decimal_x],["Round decimal on ordinate",round_decimal_y]]

    print(tabulate(table,headers=["Parameters","Input"]))
    #print(os.path.dirname(os.path.realpath(__file__)).split("/"))
    output_test_info = "output_info_"+os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]+".txt"
    #output_test_info = "output_info_"+os.path.dirname(os.path.realpath(__file__)).split("/")[-1]+".txt"
    f = open(output_test_info, "w")
    print(tabulate(table,headers=["Parameters","Input"]),file=f)
    f.close()


    return [reference_set_name,similarity_type,key_length,data_set_encoding,data_set_decoding,polynomial_degree,ecc_symbols,round_decimal_x,round_decimal_y,number_of_chaff_points,chaf_point_generator]



    


if __name__ == "__main__":

    print("Enter these informaton")
    input_info = test_info_input()

    reference_set = create_reference_set_from_file(input_info[0])
    similarity_type = input_info[1]
    key_length = input_info[2]
    data_set_encoding = input_info[3]
    data_set_decoding = input_info[4]
    polynomial_degree = input_info[5]
    ecc_symbols = input_info[6]
    round_decimal_x = input_info[7]
    round_decimal_y = input_info[8]
    number_of_chaff_points = input_info[9]
    chaf_point_generator = input_info[10]



vault,list_Xs,list_Ys,initial_polynomial,reed_solomon_redudant_coeffs,polynomial_coefficients,secret,polynomial_coefficients_hash = alice(reference_set)


#print('---------------------','Bob','STARTED','---------------------------','\n')
list_Xs_Bob = bob_first_phase(reed_solomon_redudant_coeffs)
#print('---------------------','Bob','FINISHED','---------------------------','\n')

#print('---------------------','Matching','STARTED','---------------------------','\n')
common_points = matching(vault,list_Xs_Bob)
#print('---------------------','Matching','FINISHED','---------------------------','\n')

#print('---------------------','Bob','STARTED','---------------------------','\n')
bob_second_phase(polynomial_coefficients_hash,common_points,ecc_symbols,polynomial_degree,polynomial_coefficients)
#print('---------------------','Bob','FINISHED','---------------------------','\n')


