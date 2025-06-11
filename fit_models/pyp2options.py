from packaging import version
import pandas as pd
import sys
import re

def pip2options(callscript,debug=False,warn_bad_lines=True):
    ret = dict()
    pipedir=callscript[:callscript.rfind('/')]

    if version.parse(pd.__version__) >= version.parse("1.3.0"): 
        on_bad_lines = "warn" if warn_bad_lines else "skip"
        df = pd.read_csv(pipedir + '/pip2options.sh',sep='=',header=None,comment="#",on_bad_lines=on_bad_lines)
    else:
        df = pd.read_csv(pipedir + '/pip2options.sh',sep='=',header=None,comment="#",error_bad_lines=False, warn_bad_lines=warn_bad_lines)

    for index, row in df.iterrows():
        if not pd.isnull(row.iloc[1]):
            ret[row.iloc[0]] = row.iloc[1]


    #CMD column numbers will require special parsing
    f = open(pipedir + '/pip2options.sh','r')


    #A dict() of keys and their datatypes for options that are filter dependent
    filtopts={"cmdx":int,"cmdy":int,"cmdm":int,"cmdq":int, \
                  "cmdcalmmin":float,"cmdcalmrange":float}
    
    for line in f:

        for filtopt in filtopts.keys():
            lmatch = re.match(r'^%s=\((.+)\)$' % (filtopt),line)
            #cmdy = re.match(r'^cmdy=\((.+)\)$',line)
            #cmdm = re.match(r'^cmdm=\((.+)\)$',line)

            if lmatch:
                if debug==True:
                    print(filtopt + ":",lmatch.group(1))
                lmf = re.finditer(r'\["(.+?)"\]=(.+?) ',lmatch.group(1))
                for g in lmf:
                    if debug==True:
                        print("g:",g.group(0))
                    ret[filtopt + "-" + g.group(1)] = filtopts[filtopt](g.group(2))

        """
        if cmdy:
            if debug==True:
                print("cmdy:",cmdy.group(1))
            cmdyf = re.finditer(r'\["(.+?)"\]=(\d+) ',cmdy.group(1))
            for g in cmdyf:
                if debug==True:
                    print("g:",g.group(0))
                ret["cmdy-" + g.group(1)] = int(g.group(2))

        if cmdm:
            if debug==True:
                print("cmdm:",cmdm.group(1))
            cmdmf = re.finditer(r'\["(.+?)"\]=(\d+) ',cmdm.group(1))
            for g in cmdmf:
                if debug==True:
                    print("g:",g.group(0))
                ret["cmdm-" + g.group(1)] = int(g.group(2))
         """

    f.close()

    if debug==True:
        print(ret)

    return ret

#pd.read_csv('' % (sys.))
