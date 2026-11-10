"""Verify the analyzed CSV against public UCI and Kaggle archives (standard library only)."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
from itertools import zip_longest
from pathlib import Path
import re
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
HEADER = ('age sex on_thyroxine query_on_thyroxine on_antithyroid_meds sick pregnant '
          'thyroid_surgery I131_treatment query_hypothyroid query_hyperthyroid lithium '
          'goitre tumor hypopituitary psych TSH_measured TSH T3_measured T3 TT4_measured TT4 '
          'T4U_measured T4U FTI_measured FTI TBG_measured TBG referral_source target patient_id').split()
NUMERIC = {'age','TSH','T3','TT4','T4U','FTI','TBG'}

def normalized(name, value):
    if value in ('?', ''):
        return None
    if name in NUMERIC:
        try:
            return Decimal(value)
        except InvalidOperation:
            pass  # Preserve a source token that is not a number.
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=ROOT/'thyroidDF.csv')
    parser.add_argument('--uci',type=Path,required=True)
    parser.add_argument('--kaggle',type=Path,required=True)
    args=parser.parse_args()
    local=args.csv.read_bytes()
    with ZipFile(args.kaggle) as z:
        kaggle=z.read('thyroidDF.csv')
    if kaggle != local:
        raise ValueError('CSV differs byte-for-byte from the supplied Kaggle archive')
    with ZipFile(args.uci) as z:
        raw=z.read('thyroid0387.data').decode('ascii')
    analyzed=csv.reader(io.StringIO(local.decode('utf-8-sig')))
    if next(analyzed) != HEADER:
        raise ValueError('Unexpected CSV columns or column order')
    n=0
    for n,(src,row) in enumerate(zip_longest(csv.reader(io.StringIO(raw)),analyzed),1):
        if src is None or row is None:
            raise ValueError('Different record counts')
        match=re.fullmatch(r'(.*?)\[(\d+)\]',src[-1])
        if not match:
            raise ValueError(f'Unrecognized target/identifier at record {n}')
        fields=src[:-1]+list(match.groups())
        if len(fields)!=len(row) or len(row)!=len(HEADER):
            raise ValueError(f'Unexpected field count at record {n}')
        for name,a,b in zip(HEADER,fields,row):
            if normalized(name,a)!=normalized(name,b):
                raise ValueError(f'Mismatch at record {n}, field {name}')
    if n!=9172:
        raise ValueError(f'Expected 9172 records, found {n}')
    print(f'UCI: {n} records x {len(HEADER)} fields match, including record order.')
    print('Kaggle: identical bytes.')
    print('CSV SHA-256:',hashlib.sha256(local).hexdigest())

if __name__=='__main__':
    main()
