#!/usr/bin/env python

import re

def find_table_number(fp, ind=0):
    '''find the table number from a filepath.'''
    file_name = fp.rsplit('/', 1)[-1]
    return int(re.findall('\d+', file_name)[ind])

def is_cval(fp):
    '''return true if cval.'''
    return False if fp.rsplit('/', 1)[-1].find('cval') == -1 else True

def is_pval(fp):
    '''return true if pval.'''
    return False if fp.rsplit('/', 1)[-1].find('pval') == -1 else True

def get_table_fp_with_number(tables, number):
    '''return the filepath of the table in tables that has the number.'''
    for table in tables:
        file_name = table.rsplit('/', 1)[-1]
        if find_table_number(file_name) == number:
            return table #table is the file name plus the full path
    raise ValueError('Table with number: %s not found' % number)