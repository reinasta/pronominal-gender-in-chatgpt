import numpy as np
import pandas as pd
import seaborn as sns
import csv
import re

# source: https://www.nomisweb.co.uk/datasets/aps168/reports/employment-by-occupation (UK)
occupation_stats = {'secretary': {'f_count': 533_600, 'm_count': 51_600},
                    'accountant': { 'f_count': 83_300, 'm_count': 110_100},
                    'engineer': {'f_count': 68_300, 'm_count': 475_700},
                    'technician': {'f_count': 2_297_100, 'm_count': 2_612_800},
                    'supervisor': {'f_count':162_700, 'm_count':43_400},
                    # I take worker to be an elementary occupation (see source table)
                    'worker': {'f_count': 1_376_600, 'm_count': 1_693_600},
                    'nurse': {'f_count': 494_900, 'm_count': 69_200},
                    # doctor interpreted as a health professional (see source table)
                    'doctor': {'f_count': 393_800, 'm_count': 265_800},
                    # interpreted as a Customer Service Occupation (see source table)
                    'dispatcher': {'f_count': 305_400, 'm_count':190_700},
                    'cashier': {'f_count': 783_900, 'm_count': 444_600},
                    # using a larger category as a proxy: Business, Research and Administrative
                    # Professionals (see source table)
                    'auditor': {'f_count': 390_600, 'm_count': 563_100},
                    # using a proxy category:  Health professionals n.e.c.
                    'dietitian': {'f_count': 55_000, 'm_count': 15_100},
                    # using the Artist category as a proxy (see source table)
                    'painter': {'f_count': 28_300, 'm_count': 15_500},
                    'broker': {'f_count': 9_200, 'm_count': 40_800},
                    'chef': {'f_count': 55_000, 'm_count': 140_200},
                    'firefighter': {'f_count': 4_700, 'm_count': 29_000},
                    'pharmacist': {'f_count': 39_800, 'm_count': 26_000},
                    'psychologist': {'f_count': 36_900, 'm_count': 9_500},
                    # ONS does not provide figures for female carpenters. here I'm using a figure in
                    # line with the proportion of women in the larger category of  Construction and Building Trades
                    # A figure that I found elsewhere confirms that this is a good estimate. The 2,399 figure is mentioned
                    # at https://careersmart.org.uk/occupations/equality/which-jobs-do-men-and-women-do-occupational-breakdown-gender
                    # which cites Working Futures 2021 (https://warwick.ac.uk/fac/soc/ier/researchthemesoverview/researchprojects/wf)
                    'carpenter': {'f_count': 2_620, 'm_count': 183_700},
                    'electrician': {'f_count': 4_100, 'm_count': 218_200},
                    'teacher': {'f_count': 1_130_000, 'm_count': 542_900},
                    'lawyer': {'f_count': 81_500, 'm_count': 61_100},
                    # ONS has not reliable figure for women plumbers, so I will be
                    # using the average of women employed in the larger category of Construction and Building Trades
                    'plumber': {'f_count': 1_936, 'm_count': 135_800},
                    # ONS does not provide figures for the specific category of surgeon
                    # I use specialist medical practitioner Category as a proxy; data from:
                    # https://www.statista.com/statistics/698260/registered-doctors-united-kingdom-uk-by-gender-and-specialty/
                    'surgeon': {'f_count': 39_788, 'm_count': 66_972},
                    'veterinarian': {'f_count': 11_500, 'm_count': 13_900},
                    'paramedic': {'f_count': 15_400, 'm_count': 17_300},
                    'architect': {'f_count': 4_600, 'm_count': 12_900},
                    'hairdresser': {'f_count': 208_900, 'm_count': 36_800},
                    'baker': {'f_count': 19_700	, 'm_count': 15_300},
                    'programmer': {'f_count': 70_000, 'm_count': 397_100},
                    'mechanic': {'f_count': 7_500, 'm_count': 299_000},
                    'manager': {'f_count': 1_227_500, 'm_count': 2_139_700},
                    'therapist': {'f_count': 164_100, 'm_count': 35_000},
                    'administrator': {'f_count':  1_843_100, 'm_count': 856_100},
                    'salesperson': {'f_count': 935_100, 'm_count': 612_400},
                    'receptionist': {'f_count': 171_200, 'm_count': 19_700},
                    'librarian': {'f_count': 14_000, 'm_count': 7_400},
                    }
# For clarification regarding the occupation categories see
# https://www.ilo.org/public/english/bureau/stat/isco/docs/groupdefn08.pdf
# For the US, similar data are available at:
# https://www.bls.gov/opub/reports/womens-databook/2021/home.htm

def occupation_stats_update():
    """Produces a dict of dicts representing the UK employment counts and
    percentages by gender"""
    for occ in occupation_stats.keys():
        f_count = occupation_stats[occ]['f_count']
        m_count = occupation_stats[occ]['m_count']
        occupation_stats[occ]['f_percent'] = f_count / (f_count + m_count)
        occupation_stats[occ]['m_percent'] = m_count / (f_count + m_count)
    return occupation_stats

occupations = ['technician', 'accountant', 'supervisor', 'engineer', 'worker', 'nurse',
              'dispatcher', 'cashier', 'auditor', 'dietitian', 'painter', 'broker', 'chef',
              'doctor', 'firefighter', 'secretary', 'pharmacist', 'psychologist', 'teacher',
              'lawyer', 'plumber', 'surgeon', 'veterinarian', 'paramedic', 'baker', 'programmer',
              'mechanic', 'manager', 'therapist', 'administrator', 'salesperson', 'receptionist',
              'librarian']
occupations_info = {}

data = []
with open('coref-data.txt') as text_data:
    for line in text_data:
        line.strip()
        #print(f"LINE: /{line}/")
        # pattern p matches lines with 3 sub-groups: sentence number, sentence str, score
        p = re.compile('^(\d+)\.\s*([ a-zA-Z0-9_,;\-\'"]+\.)\s*\((\d)\)')
        m = p.match(line)
        if not m:
            continue
        sentence_num = int(m.group(1))
        annotated_sentence = m.group(2)
        sentence_score = int(m.group(3))
        if sentence_num is not None and annotated_sentence and sentence_score is not None:
            datum = {}
            datum['num'] = sentence_num
            datum['sentence'] = annotated_sentence
            datum['score'] = sentence_score
            pf = re.compile('\s+(?:she|her)_')
            pm = re.compile('\s+(?:he|him|his)_')
            pn = re.compile('\s+(?:they|them|their)_')
            if pf.search(line):
                datum['gender'] = 'f'
            elif pm.search(line):
                datum['gender'] = 'm'
            elif pn.search(line):
                datum['gender'] = 'n'
            for occ in occupations:
                p = re.compile(f"{occ}")
                if p.search(line):
                    datum['occupation'] = occ
            data.append(datum)


df = pd.DataFrame(data)
#df.loc[df['num'] == 1]

def collect_occupation_info():
    """Produces a dict of dicts encoding gender employment by occupation"""
    i = 0
    for occ in occupations:
        occ_entries = df.loc[df['occupation'] == occ]
        occ_num = len(occ_entries)
        fs_with_occ = len(df.loc[(df['occupation'] == occ) & (df['gender'] == 'f') & (df['score'] == 1)])
        ms_with_occ = len(df.loc[(df['occupation'] == occ) & (df['gender'] == 'm') & (df['score'] == 1)])
        ns_with_occ = len(df.loc[(df['occupation'] == occ) & (df['gender'] == 'n') & (df['score'] == 1)])
        d = {}
        d['name'] = occ
        d['num'] = occ_num
        d['f_percent'] = fs_with_occ / occ_num
        d['m_percent'] = ms_with_occ / occ_num
        d['n_percent'] = ns_with_occ / occ_num
        # normalized difference between f% and m%
        d['fm_delta'] = (d['f_percent'] - d['m_percent']) / (d['f_percent'] + d['m_percent'])
        # f% and m% for occ in UK employment stats
        d['f_percent_uk'] = occupation_stats_update()[occ]['f_percent']
        d['m_percent_uk'] = occupation_stats_update()[occ]['m_percent']
        d['fm_delta_uk'] = (d['f_percent_uk'] - d['m_percent_uk'])
        occupations_info[i] = d
        i += 1
        #print(f"{occ}: {occ_num} ({fs_with_occ / occ_num} F, {fs_with_occ / occ_num} M)")
    return occupations_info

def coref_summary():
    f_res = df.loc[(df['gender'] == 'f') & (df['score'] == 1)]
    m_res = df.loc[(df['gender'] == 'm') & (df['score'] == 1)]
    n_res = df.loc[(df['gender'] == 'n') & (df['score'] == 1)]
    f_res_0 = df.loc[(df['gender'] == 'f') & (df['score'] == 0)]
    m_res_0 = df.loc[(df['gender'] == 'm') & (df['score'] == 0)]
    n_res_0 = df.loc[(df['gender'] == 'n') & (df['score'] == 0)]
    zero_score = df.loc[df['score'] == 0]
    one_score = df.loc[df['score'] == 1]
    f_percentage_total = (len(f_res) + len(f_res_0)) / len(df)
    f_percentage_valid = len(f_res) / len(df)
    m_percentage_total = (len(m_res) + len(m_res_0)) / len(df)
    m_percentage_valid = len(m_res) / len(df)
    n_percentage_total = (len(n_res) + len(n_res_0)) / len(df)
    n_percentage_valid = len(n_res) / len(df)
    print("COUNTS (SUMMARY)")
    print(f"F {len(f_res)} + M {len(m_res)} + N {len(n_res)} = {len(f_res) + len(m_res) + len(n_res)} (rows scored 1 per each gender)")
    print(f"F {len(f_res_0)} + M {len(m_res_0)} + N {len(n_res_0)} = {len(f_res_0) + len(m_res_0) + len(n_res_0)} (rows scored 0 per each gender)")
    print("PERCENTAGES (SUMMARY)")
    print(f"F {f_percentage_valid}, M {m_percentage_valid}, N {n_percentage_valid} (valid resolutions)")
    print(f"F {f_percentage_total}, M {m_percentage_total}, N {n_percentage_total} (all resolutions)")

def test_size():
    f_res = df.loc[(df['gender'] == 'f') & (df['score'] == 1)]
    m_res = df.loc[(df['gender'] == 'm') & (df['score'] == 1)]
    n_res = df.loc[(df['gender'] == 'n') & (df['score'] == 1)]
    zero_score = df.loc[df['score'] == 0]
    one_score = df.loc[df['score'] == 1]
    if len(f_res) + len(m_res) + len(n_res) + len(zero_score) == len(df):
        print(f"\033[32;1mData frame integrity OK.\033[0m")
    else:
        print(f"\033[31;1mYou failed to parse some sentences in your data.\033[0m")
        print(f"{len(f_res) + len(m_res) + len(n_res)} (rows scored 1 actually processed).")
        print(f"{len(zero_score)} (total rows scored 0)")
        print(f"{len(one_score)} (total rows scored 1)")
        print(f"{len(df)} (total rows in data frame)")
        #compute: df - f_res - m_res - n_res - zero_score
        fs = f_res['num'].tolist()
        ms = m_res['num'].tolist()
        ns = n_res['num'].tolist()
        zs = zero_score['num'].tolist()
        rest = set(df['num'].tolist()) - set(fs) - set(ms) - set(ns) - set(zs)
        rest = list(rest)
        rest.sort()
        print(f"Missing rows for sentences numbered:\n", rest)



test_size()
coref_summary()
oinfo = collect_occupation_info()

# using dictionary to convert specific columns  
convert_dict = {'num': int,
                #'f_count': int,
                #'m_count': int,
                'f_percent': float,
                'm_percent': float,
                'n_percent': float,
                'fm_delta': float,
                'fm_delta_uk': float,
                'f_percent_uk': float,
                'm_percent_uk': float}

#df = df.astype(convert_dict)

occ_df = pd.DataFrame(oinfo).transpose()
occ_stats_df = pd.DataFrame(occupation_stats_update()).transpose()
occ_df = occ_df.astype(convert_dict)
#print(occ_df.dtypes)
occ_sorted = occ_df.sort_values(by='fm_delta', ascending=False)

def get_bergsma_data():
    bergsma_data = []
    with open('occupations-stats.tsv') as csv_file:
        csv_data = csv.reader(csv_file, delimiter='\t')
        # row shape: ['occupation', 'bergsma_pct_female', 'bls_pct_female', 'bls_year']
        for row in csv_data:
            d = {}
            if row[0] in occupations:
                d['name'] = row[0] # occupation name
                d['f_percent_bergsma'] = float(row[1]) / 100
                # f + m = 100 => f - m = 100 - 2m = 100 - 2 (100 - f) = -100 + 2f
                # => f - m = 2f - 100
                d['fm_delta_bergsma'] = 2 * d['f_percent_bergsma'] - 1.00
                bergsma_data.append(d)
    return bergsma_data

bergsma_data_df = pd.DataFrame(get_bergsma_data())
bergsma_data_df = bergsma_data_df.astype({"name": str, "f_percent_bergsma": float})

# Plot chatgpt coreference resolution data vs uk employment data
occ_diffs_chatgpt_df = occ_df.loc[:, ["name", "fm_delta"]]
occ_diffs_onsuk_df = occ_df.loc[:, ["name", "fm_delta_uk"]]
occ_diffs_bergsma_df = bergsma_data_df.loc[:, ["name", "fm_delta_bergsma"]]
# rename column so both dfs have the same column names (used to concatenate dfs)
occ_diffs_onsuk_df = occ_diffs_onsuk_df.rename(columns={"fm_delta_uk": "fm_delta"})
occ_diffs_bergsma_df = occ_diffs_bergsma_df.rename(columns={"fm_delta_bergsma": "fm_delta"})
# build lists to be used as category columns
category_col_chatgpt = ['chatgpt'] * len(occ_df)
category_col_onsuk = ['onsuk'] * len(occ_df)
category_col_text = ['text'] * len(occ_df)
# first df: chatgpt data
occ_diffs_chatgpt_df["category"] = category_col_chatgpt
occ_diffs_chatgpt_df["f_stats_uk"] = occ_df["f_percent_uk"]  # stats_uk is used for the x axis
# second df: ons uk data
occ_diffs_onsuk_df["category"] = category_col_onsuk
occ_diffs_onsuk_df["f_stats_uk"] = occ_df["f_percent_uk"]  # stats_uk is used for the x axis
# third df: text data (from bergsma)
occ_diffs_bergsma_df["category"] = category_col_text
occ_diffs_bergsma_df["f_stats_uk"] = occ_df["f_percent_uk"]  # stats_uk is used for the x axis
# concatenate the three dfs
occ_diffs = pd.concat([occ_diffs_chatgpt_df, occ_diffs_onsuk_df, occ_diffs_bergsma_df], ignore_index=True)
#occ_diffs

sns.set_style('darkgrid')
plt = sns.lmplot(data=occ_diffs, x='f_stats_uk', y='fm_delta', hue='category', legend=False)
plt.set(xlabel='% of women by occupation in the UK (ONS 2021)', ylabel='% differential (women - men) by occupation')
plt.set(title="Gender bias in ChatGPT's coreference resolution")
plt.set(ylim=(-1.0, 1.0))
plt.axes[0,0].legend(loc='upper left', title='Category')

def label_point(x, y, val, ax):
    ax = ax.axes[0,0]
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        if point['val'] in ['plumber', 'worker', 'technician', 'nurse', 'secretary']:
            if point['val'] in ['worker', 'nurse']:
                ax.text(point['x']-.02, point['y']+.05, str(point['val']))
            elif point['val'] == 'secretary':
                ax.text(point['x']-.02, point['y']-.08, str(point['val']))
            else: # plumber
                ax.text(point['x']+.02, point['y']-.04, str(point['val']))

label_point(occ_df.f_percent_uk, occ_df.fm_delta, occ_df.name, plt)


print(f"BERGSMA\n", bergsma_data_df)
#occ_diffs
print(occ_diffs.to_string())
#bergsma_data_df

#occ_stats_df
#occ_df
#occ_sorted

#print(occupations_info)
