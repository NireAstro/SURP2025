# ???
ogle=~/ogle
nfmax=100

##### DIRECTORIES #####
# PROGRAM LOCATIONS
fitshdir=/usr/local/bin
gnuplot=/usr/bin/gnuplot
ds9=/usr/local/bin/ds9
isisdir=$pipedir/isis/decam
isisbin=$pipedir/isis/bin

# INPUT DATA LOCATIONS
imagedir=/fs/project/gaudi.1/Erin/MISHAPS/images
refdir=/fs/project/gaudi.1/Erin/MISHAPS/ref
lcdir=/fs/project/gaudi.1/Erin/MISHAPS/lc
isodir=/fs/project/gaudi.1/Erin/MISHAPS/isochrones
modeldir=/fs/project/gaudi.1/Erin/MISHAPS/JKTEBOP_Models
refimage=ref.fits
md5sum=md5sum.csv

# WORKING DIRECTORIES
workdir=/home/ali/work_dir
lcworkdir=/home/ali/work_dir

# DATA PRODUCT LOCATIONS
outdir=/fs/project/gaudi.1/Erin/MISHAPS/processed
maskdir=/work/acrisp3/mishaps/ref/mask_files
canddir=/fs/project/gaudi.1/Erin/MISHAPS/candidates
gaiadir=/fs/project/gaudi.1/Erin/MISHAPS/gaia_query
rmsdir=/fs/project/gaudi.1/Erin/MISHAPS/rms_plots
seddir=/fs/project/gaudi.1/Erin/MISHAPS/sed_fits

##### HPC-RELATED #####
# PBS VARS
allocation=hpc_mishaps03

# MODULES
pythonmodule=python/3.8.5-anaconda
intelmodule=intel/19.0.5

##### INSTRUMENT/OBSERVATION INFO #####
instrument=DECam
latitude=-70.81489d
longitude=-30.16606d
altitude=2215.0m 
goodchips=MISHAPS_goodchips.txt
nfilters=2
goodfilters=MISHAPS_goodfilters.txt

##### COLUMN NUMBERS #####
# LIGHTCURVE COLS (original + detrended)
tcol=12 
magcol=15
errcol=16
dtrtcol=1
dtrmagcol=2
dtrerrcol=3

# CMD COLS
cmdracol=2
cmddeccol=3

declare -A cmdx
cmdx=( ["r"]=16 ["VR"]=27 ["z"]=5 )

declare -A cmdy
cmdy=( ["r"]=17 ["VR"]=28 ["z"]=6 )

declare -A cmdm
cmdm=( ["r"]=18 ["VR"]=29 ["z"]=7 )

declare -A cmdq
cmdq=( ["r"]=21 ["VR"]=32 ["z"]=10 )

##### SCRIPT VARIABLES #####
# CALPHOT
# Minimum magnitude for DIA-PSF photometry calibration
declare -A cmdcalmmin
cmdcalmmin=( ["r"]=16.5 ["VR"]=17.0 ["z"]=17.2 )

declare -A cmdcalmrange
cmdcalmrange=( ["r"]=16.5 ["VR"]=16.5 ["z"]=16.5 )

# TRANSIT SEARCH
# Cuts for transit search
snrcut=7.0
depthcut=-0.1

##### MISC #####
usepigz=16
