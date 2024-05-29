#!/bin/bash

# Aligned XSum
./xsum_general.sh \
    /local1/hfs/gs_stuff/al-xsum-bild-res/0-6_2-0 \
    paulh27/xsum_aligned_smallmT5 \
    0.6 \
    2.0

./xsum_general.sh \
    /local1/hfs/gs_stuff/al-xsum-bild-res/0-4_5-0 \
    paulh27/xsum_aligned_smallmT5 \
    0.4 \
    5.0

# Unaligned XSum
./xsum_general.sh \
    /local1/hfs/gs_stuff/ft-xsum-bild-res/0-5_3-0 \
    lilferrit/ft-xsum \
    0.5 \
    3.0

./xsum_general.sh \
    /local1/hfs/gs_stuff/ft-xsum-bild-res/0-3_5-0 \
    lilferrit/ft-xsum \
    0.3 \
    5.0

# Aligned WMT14
./wmt14_general.sh \
    /local1/hfs/gs_stuff/al-wmt14-bild-res/0-8_2-0 \
    paulh27/wmt_aligned_smallT5_cont0 \
    0.8 \
    2.0

./wmt14_general.sh \
    /local1/hfs/gs_stuff/al-wmt14-bild-res/0-5_3-0 \
    paulh27/wmt_aligned_smallT5_cont0 \
    0.5 \
    3.0

# TODO: WMT14 Unaligned (fine-tuned)