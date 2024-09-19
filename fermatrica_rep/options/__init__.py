"""
All concerning option calculations:

1. Prepare data and settings in `fermatrica_rep.options.define`
2. Set translators in `fermatrica_rep.options.translators`
3. Calculate single option in `fermatrica_rep.options.calc`
4. Calculate multiple options without optimization in `fermatrica_rep.options.calc_multi`
5. Optimize budget split in `fermatrica_rep.options.optim`
"""


import fermatrica_rep.options.calc
import fermatrica_rep.options.calc_multi
import fermatrica_rep.options.define
import fermatrica_rep.options.optim
import fermatrica_rep.options.translators
