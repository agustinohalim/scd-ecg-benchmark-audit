"""Download the PhysioNet data the reanalysis needs (open access, ODC-By 1.0).

    sddb   MIT-BIH Sudden Cardiac Death Holter Database, all files (about 1.3 GB)
    nsrdb  MIT-BIH Normal Sinus Rhythm Database, all files (about 0.5 GB)
    chfdb  BIDMC Congestive Heart Failure Database, headers and beat annotations only; the
           3-minute signal segments are streamed from PhysioNet by 02_citra.py

Also copies `VF Data.csv` (VF onset per sddb record, from the 2022 thesis) next to the data.

Target folder: $THESIS_DATA, or reanalysis_2026/data/physionet by default.
Usage:  python 00_unduh_data.py
"""
import os
import shutil

import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('THESIS_DATA', os.path.join(HERE, 'data', 'physionet'))


def main():
    for db in ('sddb', 'nsrdb'):
        tujuan = os.path.join(DATA, db)
        os.makedirs(tujuan, exist_ok=True)
        wfdb.dl_database(db, tujuan)
        wfdb.dl_files(db, tujuan, ['RECORDS'])
        print(db, 'done', flush=True)
    chf = os.path.join(DATA, 'chfdb')
    os.makedirs(chf, exist_ok=True)
    wfdb.dl_files('chfdb', chf, [f'chf{i:02d}.{e}' for i in range(1, 16) for e in ('hea', 'ecg')])
    shutil.copy(os.path.join(HERE, '..', 'original_2022', 'VF Data.csv'), DATA)
    print('data in', DATA)


if __name__ == '__main__':
    main()
