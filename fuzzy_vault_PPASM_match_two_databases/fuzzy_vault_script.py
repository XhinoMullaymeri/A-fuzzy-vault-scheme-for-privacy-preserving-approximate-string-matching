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

def read_vault_from_file(file_name):
    with open(file_name) as file: 
        lines = [line.strip() for line in file.readlines()]
        vault = lines[1]
        vault = vault[1:-1]
        vault = vault.split(",")
        
        for index , value in enumerate(vault):
            if index%2 == 0:
                vault[index] = value.strip()[1:]
            else:
                vault[index] = value.strip()[:-1]

        new_vault = []
        for index , value in enumerate(vault):
            if index%2 == 1:
                new_vault.append((float(vault[index-1]),float(value)))
        
        return(new_vault)

def create_secret(**kwargs):
    """
    This function creates the secret
    Params: none
    Returns: string secret
    """
    #secret= input('Secret is: ')
    
    #All uppercase English letters in a lsit
    alphabet = list(string.ascii_uppercase)

    #default n is set to n = 10
    n = kwargs.get('k', 10)

    #creating a random secret
    secret = ''
    for _ in range(n):
        secret += random.choice(alphabet)

    return secret


def encrypt_secret(secret):
    """
    This function encrypts the secret and returns encrypted digest
    Params: string secret
    Returns: string encrypted_hexa
    """
    encrypted_secret = hashlib.sha1(secret.encode()) 
    encrypted_hexa = encrypted_secret.hexdigest()
    #print('length: ',len(encrypted_hexa),'\n','key(hexa): ',encrypted_secret)
    return encrypted_secret.digest()


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

    

def encrypt_key(key):
    """
    This function has as input vault's key i.e. a name and encrypts it.
    Params: string key
    Retuns: string encrypted_hexa
    """
    encrypted_key = hashlib.sha1(key.encode()) 
    encrypted_hexa = encrypted_key.hexdigest()
    #print('length: ',len(encrypted_hexa),'\n','key(hexa): ',encrypted_key)
    return encrypted_key.digest()


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

    # if input('Show fuzzy vault plot?')=='y':
    #     fig, ax = plt.subplots()
    #     ax.set_title('{!r}'.format('Fuzzy Vault'), color='C0')
    #     ax.plot(sample_x, sample_y,'.',label='CHAFF POINTS',color ='red',ms = 15 )
    #     ax.plot(list(unzipped[0]),list(unzipped[1]),'.',label='GENUINE POINTS',color='green',ms = 15)
    #     ax.legend()


    #     # x = np.linspace(-1, 1, 50)
    #     # y = [initial_polynomial(i) for i in x]
    #     # ax.plot(x, y)



    #     fig.show()




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
    
    hashes = {}
    for tupple in distances:
        hashes[tupple[0]] = dummy_word_hash(tupple[0])

    hashes = [ dummy_word_hash(tupple[0]) for tupple in distances]
    return hashes[:key_length]

def alice(secret):
    
    #creating the secret
    #secret = create_secret(k = 4)   #random secret
    
    #secret = input('secret: ').upper()
    #if input('Secret created...') == 'y':
    #    print(secret)
    
    t_start_enc = time.time()

    #creating the key
    key = secret_to_key(secret,key_length,reference_set)
    #print(key)
    t_key_created = time.time()
    
    #rsc codec 
    reed_solomon_redudant_coeffs=[]
    if ecc_symbols > 0:
        rsc = reedsolo.RSCodec(ecc_symbols)
        #encode secret UTF-8 first we make string to bytes
        enc_secret = rsc.encode(secret.encode())
        reed_solomon_redudant_coeffs=[item for item in enc_secret][-ecc_symbols:]
        msg = "Secret encoded in rs("+str(ecc_symbols)+"+k,k)..."
    t_RS_created = time.time()
   
    
    # if input(msg) =='y':
    #     print(enc_secret)
    #     print([item for item in enc_secret])


    #Creating polynomial
    #getting coeffs from secret (ascii numbers) 
    polynomial_coefficients = secret_to_polynomial_coefficients(secret,mode=3,degree=polynomial_degree)
    #polynomial_coefficients_hash = round(math.cos(sum(polynomial_coefficients)),round_decimal_x)
    concatenated_coeffs = "".join(str(coeff) for coeff in polynomial_coefficients)
    polynomial_coefficients_hash = hashlib.md5(concatenated_coeffs.encode()).hexdigest() 
    #print(polynomial_coefficients_hash)
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
    list_Xs = list(set(list_Xs))
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

    t_genuine_created =  time.time()

    if chaf_point_generator == 0:
        vault = add_chaf_points(vault,number_of_chaff_points)
    
    if chaf_point_generator == 1:
        vault = add_chaf_points_using_reference_set(vault,number_of_chaff_points,reference_set,ref_chaff_perc,initial_polynomial)

    t_chaff_created =  time.time()

    #if input('Vault added chaf points...')=='y':
    #    print(vault)

    random.shuffle(vault)

    t_shuffled  =  time.time()
    #if input('Vault shuffled...')=='y':
    #    print(vault)
    times =[t_key_created-t_start_enc,t_RS_created-t_key_created,t_genuine_created-t_RS_created,t_chaff_created-t_genuine_created,t_shuffled-t_chaff_created]
    times.append(sum(times))
    times_for_encoding.append(times)
    return vault,list_Xs,list_Ys,initial_polynomial,reed_solomon_redudant_coeffs,polynomial_coefficients,secret,polynomial_coefficients_hash


def bob_first_phase(bob_key,reed_solomon_redudant_coeffs,times_bob_phase):
    t_bob_started = time.time()

    bob_key = bob_key.upper()
    temp1 = bob_key
    Bob_list_x = secret_to_key(bob_key,key_length,reference_set)

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
    Bob_list_x = list(set(Bob_list_x))
    #print("std dev: ",statistics.stdev(Bob_list_x))

    t_bob_key_created = time.time()
    if ecc_symbols > 0:
        try:
            rsc = reedsolo.RSCodec(ecc_symbols)
            codeword = [coeff for coeff in bob_key.encode()] + reed_solomon_redudant_coeffs
            dec_seret = rsc.decode(codeword)
            bob_key = dec_seret[0].decode("utf-8")
            Bob_list_x = secret_to_key(bob_key,key_length,reference_set)
            temp=[]
            for x in Bob_list_x:
                random.seed(a=x)
                temp.append(random.randint(0,1000000))
            Bob_list_x = temp
            Bob_list_x =[math.cos(x) for x in Bob_list_x]
            Bob_list_x = [round(x,round_decimal_x) for x in Bob_list_x]
            Bob_list_x = list(set(Bob_list_x))
            # if bob_key != temp1:
            #     number = number +1 
            #     print(temp1, bob_key)
            #     #print(len(temp1),len(bob_key))
            #     #print([c for c in temp1])
            #     #print([c for c in bob_key])
            # else:
            #     print(bob_key,temp1)

        except:
            pass
    # finally:
    #     t_rs_decoding = time.time()
    t_rs_decoding = time.time()
    times_bob_phase.append(t_bob_key_created-t_bob_started)
    times_bob_phase.append(t_rs_decoding-t_bob_key_created)

    return  Bob_list_x,times_bob_phase


def matching(vault,Bob_list_x,times_bob_phase):
    t_common_start = time.time()
    unzipped_vault = list(zip(*vault))
    vault_x = unzipped_vault[0]
    vault_y = unzipped_vault[1]

    common_points=[]

    for index, value in enumerate (vault_x):
        if value in Bob_list_x:
            common_points.append((value,vault_y[index]))
    
    t_common_end = time.time()
    times_bob_phase.append(t_common_end-t_common_start)

    return common_points,times_bob_phase


def bob_second_phase(bob_key,common_points,rsc,polynomial_degree,polynomial_coefficients,index_dec,times_bob_phase,polynomial_coefficients_hash):
    global names_that_matched

    # t_combinations_created = time.time()
    # if len(common_points) >=9:
    #     t_bob_second_started = time.time()

    #     unzipped = list(zip(*common_points))
    #     reconstructed_polynomial_lagrange = lagrange(list(unzipped[0]), list(unzipped[1]))
    #     reconstructed_polynomial_polyfit = np.polyfit(list(unzipped[0]), list(unzipped[1]), 8)

    #     reconstructed_polynomial_coefficients=[]
    #     for coeff in reconstructed_polynomial_lagrange.coefficients:
    #         coeff = round(coeff)
    #         coeff = int(coeff)
    #         reconstructed_polynomial_coefficients.append(coeff)

    #     first_usefull_coeff = next((index for index,value in enumerate(reconstructed_polynomial_coefficients) if value != 0), -1)
    #     usefull_coeffs = reconstructed_polynomial_coefficients[-(len(reconstructed_polynomial_coefficients)-first_usefull_coeff):]
        

    #     diff_lagrange = [abs(x1 - x2) for (x1, x2) in zip(usefull_coeffs, polynomial_coefficients)]


    #     reconstructed_polynomial_coefficients= []
    #     for coeff in reconstructed_polynomial_polyfit:
    #         coeff = round(coeff)
    #         coeff = int(coeff)
    #         reconstructed_polynomial_coefficients.append(coeff)

    #     diff_polyfit = [abs(x1 - x2) for (x1, x2) in zip(reconstructed_polynomial_coefficients, polynomial_coefficients)]
        
    #     if sum(diff_lagrange) == 0 or sum(diff_polyfit) ==0:
    #         names_that_matched.append((ncids_for_decoding[index_dec],bob_key,len(common_points)))
    #         times_bob_phase = ["is_key"+str(len(common_points))] + times_bob_phase
             

    #     t_combinations_done = time.time()

    #     if sum(diff_lagrange) != sum(diff_polyfit):
    #         print("real poly coeffs:",polynomial_coefficients)
    #         print("lagrange:",usefull_coeffs)
    #         print(diff_lagrange)
    #         print("polyfit:",reconstructed_polynomial_coefficients)
    #         print(diff_polyfit)





    #     times_bob_phase.append(t_combinations_done-t_combinations_created)
    #     return times_bob_phase

    combinations = list(itertools.combinations(common_points,polynomial_degree+1))
    
    t_combinations_created = time.time()
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
        #diff = [abs(x1 - x2) for (x1, x2) in zip(usefull_coeffs, polynomial_coefficients)]
            

        interpolated_polynomial_coefficients_hash = "".join( str(coeff) for coeff in usefull_coeffs)
        interpolated_polynomial_coefficients_hash = hashlib.md5(interpolated_polynomial_coefficients_hash.encode()).hexdigest()
        if polynomial_coefficients_hash == interpolated_polynomial_coefficients_hash:
            # print(polynomial_coefficients)
            # print(reconstructed_polynomial_coefficients)
           #print("matched with:",bob_key,"\tCommon points:",len(common_points))
            names_that_matched.append((ncids_for_decoding[index_dec],bob_key,len(common_points)))
            times_bob_phase = ["is_key"+str(len(common_points))] + times_bob_phase
            break


    t_combinations_done= time.time()

    times_bob_phase.append(t_combinations_done-t_combinations_created)
    return times_bob_phase



#GLOBALS   
number_of_chaff_points = 150
ecc_symbols = 0
polynomial_degree = 8
key_length = 15
ref_chaff_perc = 1
round_decimal_x = 6
round_decimal_y = 8
chaf_point_generator = 1
names_that_matched=[]
dictionary_with_names_and_matched_names = {}
similarity_type =""
interpolation_method = "Lagrange interpolation"
#time for key to be created (key from secret and reference set)
#time for RS redudant
#time for genuine projection --> fuzzy vault
#time for adding chaff points --> fuzzy vault
#time for shuffling --> fuzzy vault
#total time
times_for_encoding=[]
len_names_for_decoding = 0

#time for key to be created
#time for RS decoding
#time for common points
#time for combinations
#total time
times_for_decoding=[]


def second_to_h_m_s(seconds):
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds) 

def estimated_time(len_enc,len_dec,times_for_encoding,times_for_decoding,times,include_times,index_enc):

    temp_enc = times_for_encoding[include_times*(times-1):include_times*times]
    temp_dec = times_for_decoding[include_times*(times-1):include_times*times]

    #print(temp_dec)
    total_enc = sum([time[len(times_for_encoding[0])-1] for time in temp_enc])
    total_dec = sum([time[len(times_for_decoding[0])-1] for time in temp_dec])

    print("avg time (last",include_times,"):",(total_enc+total_dec)/include_times)
    print("tested:",index_enc,"tests left:",(len_enc*len_dec))
    seconds = int(((total_enc+total_dec)/include_times) * len_enc * len_dec)
    return second_to_h_m_s(seconds)


def test_info_input():

    #reference set 
    # index 0 has reference set number and #index 0 has its type
    reference_set_params = [2,25]
    #reference_set_params  = input("Reference set params... e.g.  1,20: ").split(',')
    #reference_set_name = 'reference_set'+str(reference_set_params[0])+'_'+str(reference_set_params[1])+'x26.txt'
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

    data_set_encoding = input("Enter encoding data set's name: ")
    data_set_decoding = input("Enter decoding data set's name: ")
    #data_set_encoding = "A_1k_names_combined"
    #data_set_decoding = "B_1k_names_combined"

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
    #output_test_info = "output_info_"+os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]+".txt"
    output_test_info = "output_info_"+os.path.dirname(os.path.realpath(__file__)).split("/")[-1]+".txt"
    f = open(output_test_info, "w")
    print(tabulate(table,headers=["Parameters","Input"]),file=f)
    f.close()


    return [reference_set_name,similarity_type,key_length,data_set_encoding,data_set_decoding,polynomial_degree,ecc_symbols,round_decimal_x,round_decimal_y,number_of_chaff_points,chaf_point_generator]



def write_output_files(ncids_for_encoding,ncids_for_decoding):
    global len_names_for_decoding
    output_file_name = "output_"+os.path.dirname(os.path.realpath(__file__)).split("/")[-1]+".txt"
    output_file_times = "output_times_"+os.path.dirname(os.path.realpath(__file__)).split("/")[-1]+".txt"

    f = open(output_file_name, "w",encoding="utf-8")
    for x, y in dictionary_with_names_and_matched_names.items():
        print(x, y,file=f)
    f.close()


    total_enc = sum([time[len(times_for_encoding[0])-1] for time in times_for_encoding])
    total_dec = sum([time[len(times_for_decoding[0])-1] for time in times_for_decoding])

    f = open(output_file_times, "w",encoding="utf-8")
    print("Total encoding time: ",second_to_h_m_s(int(total_enc)),"\n","Total decoding time: ",second_to_h_m_s(int(total_dec)),file=f)
    print("\n\nTIMES FOR ENCODING\n\n",file=f)

    for index_t_enc,times in enumerate (times_for_encoding):
        print(ncids_for_encoding[index_t_enc],":\t",times,file=f)
    
    print("\n\nTIMES FOR DECODING\n\n",file=f)
    for index_t_enc_2,ncid in enumerate (ncids_for_encoding):
        print("\t\t\t secret ncid:",ncid,file=f)
        for index_t_dec,ncid_dec in enumerate(ncids_for_decoding):
            #print(len(times_for_decoding),index_t_enc_2*len_names_for_decoding+index_t_dec,index_t_enc_2,len_names_for_decoding,index_t_dec)
            print(ncid_dec,":\t",times_for_decoding[index_t_enc_2*len_names_for_decoding+index_t_dec],file=f)
    f.close()


    


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

    #input("\nPress any key to start the test...")
    #print("started...")

#names for Alice encoding
    with open(data_set_encoding, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        names = list(csv_reader)
        ncids_for_encoding = [i[0] for i in names] 
        names_for_encoding = [i[1] for i in names] 


#names for decoding
    with open(data_set_decoding, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        names = list(csv_reader)
        ncids_for_decoding = [i[0] for i in names] 
        names_for_decoding = [i[1] for i in names] 


    len_names_for_decoding = len(names_for_decoding) - 1


    #print(names_for_encoding)
    #names_for_decoding_ALL = names_for_decoding + names_for_decoding

    times = 1
    for index_enc,name in enumerate (names_for_encoding[:2]):
        vault,list_Xs,list_Ys,initial_polynomial,reed_solomon_redudant_coeffs,polynomial_coefficients,secret,polynomial_coefficients_hash = alice(name)
        names_that_matched=[]
        if  (index_enc%100==0 and  index_enc > 1) :
            print("Estimated time:", estimated_time(len(names_for_encoding)-index_enc,len(names_for_decoding),times_for_encoding,times_for_decoding,times,10000,index_enc))
            times=times+1

        for index_dec,bob_key in enumerate (names_for_decoding[:1]):
            
            #print(name,bob_key)
            times_bob_phase=[]
            #print('---------------------','Bob','STARTED','---------------------------','\n')
            list_Xs_Bob,times_bob_phase = bob_first_phase(bob_key,reed_solomon_redudant_coeffs,times_bob_phase)
            #print('---------------------','Bob','FINISHED','---------------------------','\n')

            #print('---------------------','Matching','STARTED','---------------------------','\n')
            common_points,times_bob_phase = matching(vault,list_Xs_Bob,times_bob_phase)
            #print('---------------------','Matching','FINISHED','---------------------------','\n')

            #print('---------------------','Bob','STARTED','---------------------------','\n')
            times_bob_phase = bob_second_phase(bob_key,common_points,ecc_symbols,polynomial_degree,polynomial_coefficients,index_dec,times_bob_phase,polynomial_coefficients_hash)
            #print('---------------------','Bob','FINISHED','---------------------------','\n')

            if isinstance(times_bob_phase[0],float):  #"is_key"
                times_bob_phase = ["not_key"] + times_bob_phase
            times_bob_phase.append(sum(times_bob_phase[1:]))
            times_for_decoding.append(times_bob_phase)

        dictionary_with_names_and_matched_names[ncids_for_encoding[index_enc]] = [name,names_that_matched]
        #print("num:",index_enc,"key:",name,"ncid:",ncids_for_encoding[index_enc],"\n unlocked:",names_that_matched) 





    total_enc = sum([time[len(times_for_encoding[0])-1] for time in times_for_encoding])
    total_dec = sum([time[len(times_for_decoding[0])-1] for time in times_for_decoding])
    print(total_enc,"\n",total_dec)

    write_output_files(ncids_for_encoding,ncids_for_decoding)

    #print("total:",sum(times_for_encoding[:][len(times_for_encoding-1)]) + sum(times_for_decoding[:][len(times_for_decoding)-1]))

    #running_tests_list_of_keys(A_names)
    
    #running_tests_random_key_generator(1)

    #tests  = int(input("How many tests: "))
    #start = time.time()
    
    #running_tests_random_key_generator(tests,7,18)

    #end = time.time()
    #print("time:",end - start)

