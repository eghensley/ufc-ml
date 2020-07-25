#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:28:04 2020

@author: ehens86
"""

from training import tune_ml, fill_ml_training_scores, optimize_bet, predict_bet_winners
from elo import optimize_elo, populate_elo
from db import pop_future_bouts, pop_year_bouts, update_mybookie

optimize_bet(trials = 5000)

#pop_year_bouts()



#import argparse
## fix ufc odds from 6-6
#FIGHT_ID = '1e13936d708bcff7'
#
#valid_pos_args = ['TRAIN', 'FULL_TRAIN', 'DB', 'PRED']
#valid_domains = ['STRIKE', 'GRAPP', 'KO', 'SUB', 'ALL', 'BET', 'ODDS']
#valid_models = ['LOG', 'LIGHT', 'DART']
#valid_dims = ['ML', 'ELO', 'BET']
#
#def db_update(args):
#    if args.domain is None:
#        if args.year is None:
#            year = 2020
#        else:
#            year = args.year
#            
#        if args.fut is None:
#            future = False
#        else:
#            future = args.fut
#            
#        if future:
#            pop_future_bouts()
#        else:
#            pop_year_bouts(year)
#    elif args.domain.upper() == 'ODDS':
#        update_mybookie(FIGHT_ID)
#        
#def bet_evaluation(args):
#    predict_bet_winners(FIGHT_ID)
#    
#def full_train(args):
#    
#    if args.iter is None:
#        n_iter = 5000
#    else:
#        n_iter = int(args.iter)
#        
#    if args.refit is None:
#        refit = False
#    else:
#        refit = args.refit
#
#    tune_ml(clf = 'log', domain = 'strike', refit = refit, post_feat = False, opt = False, trials = n_iter)
#    tune_ml(clf = 'log', domain = 'grapp', refit = refit, post_feat = False, opt = False,  trials = n_iter)
#    tune_ml(clf = 'light', domain = 'strike', refit = refit, post_feat = False, opt = False,  trials = n_iter)
#    tune_ml(clf = 'light', domain = 'grapp', refit = refit, post_feat = False, opt = False,  trials = n_iter)    
#    
#    tune_ml(clf = 'log', domain = 'strike', refit = refit, post_feat = True, opt = False,  trials = n_iter)
#    tune_ml(clf = 'log', domain = 'grapp', refit = refit, post_feat = True, opt = False,  trials = n_iter)
#    tune_ml(clf = 'light', domain = 'strike', refit = refit, post_feat = True, opt = True,  trials = int(n_iter/4))
#    tune_ml(clf = 'light', domain = 'grapp', refit = refit, post_feat = True, opt = True,  trials = int(n_iter/4))       
#    tune_ml(clf = 'dart', domain = 'strike', refit = refit, post_feat = True, opt = True,  trials = int(n_iter/4))
#    tune_ml(clf = 'dart', domain = 'grapp', refit = refit, post_feat = True, opt = True,  trials = int(n_iter/4))   
#    
#    fill_ml_training_scores()
#    
#    optimize_elo('strike', trials = int(n_iter/10))
#    optimize_elo('grapp', trials = int(n_iter/10))
#    optimize_elo('ko', trials = int(n_iter/10))
#    optimize_elo('sub', trials = int(n_iter/10))
#
#    populate_elo(refit = True)
#    
#    tune_ml(clf = 'log', domain = 'all', refit = refit, post_feat = False, opt = False,  trials = n_iter)
#    tune_ml(clf = 'light', domain = 'all', refit = refit, post_feat = False, opt = False,  trials = int(n_iter/2))
#    tune_ml(clf = 'dart', domain = 'all', refit = refit, post_feat = False, opt = False,  trials = n_iter)
##    
#    tune_ml(clf = 'log', domain = 'all', refit = refit, post_feat = True, opt = False,  trials = n_iter)
#    tune_ml(clf = 'light', domain = 'all', refit = refit, post_feat = True, opt = True,  trials = int(n_iter/4))
#    tune_ml(clf = 'dart', domain = 'all', refit = refit, post_feat = True, opt = True,  trials = int(n_iter/4))
#
#def train(args):
#    funct_domain = 'strike'
#    funct_model = 'log'
#
#    if args.iter is None:
#        n_iter = 5000
#    else:
#        n_iter = int(args.iter)
#        
#    if args.domain is None:
#        print("No domain specified.. defaulting to STRIKE")
#    elif args.domain.upper() not in valid_domains:
#        raise ValueError("%s is not a valid domain" % (args.domain))
#    else:
#        funct_domain = args.domain.lower()
#        
#    if args.db and args.refit:
#        fill_ml_training_scores()
#    else:
#        if args.dim.upper() == 'ELO':
#            optimize_elo(funct_domain)
#        elif args.dim.upper() == 'BET':
#            optimize_bet(trials = n_iter)
#        else:        
#            if args.model is None:
#                print("No model specified.. defaulting to log")
#            elif args.model.upper() not in valid_models:
#                raise ValueError("%s is not a valid domain" % (args.domain))
#            else:
#                funct_model = args.model.lower()
#                
#            tune_ml(clf = funct_model, domain = funct_domain, refit = args.refit, post_feat = args.post, opt = args.opt)
#
#if __name__ == '__main__':
#    # Instantiate the parser
#    parser = argparse.ArgumentParser(description='UFC Prediction Engine')
#    # Required positional argument
#    parser.add_argument('pos_arg', type=str,
#                        help='Function to trigger.  Valid options: %s' % [", ".join(valid_pos_args)])
#    
#    # Optional positional argument
#    parser.add_argument('opt_pos_arg', type=int, nargs='?',
#                        help='An optional integer positional argument')
#    
#    # Optional argument
#    parser.add_argument('--domain', type=str,
#                        help='Domain to activate.  Valid options: %s ' % [", ".join(valid_domains)])
#
#    # Optional argument
#    parser.add_argument('--model', type=str,
#                        help='Model to activate.  Valid options: %s ' % [", ".join(valid_models)])
#
#    # Optional argument
#    parser.add_argument('--dim', type=str,
#                        help='Dimension to activate.  Valid options: %s ' % [", ".join(valid_dims)])
#
#    # Optional argument
#    parser.add_argument('--iter', type=int,
#                        help='Number of iterations')
#    
#    # Optional argument
#    parser.add_argument('--year', type=int,
#                        help='Year to use')
#    
#    # Switch
#    parser.add_argument('--refit', action='store_true',
#                        help='A boolean switch')
#    
#    # Switch
#    parser.add_argument('--post', action='store_true',
#                        help='A boolean switch')
#
#    # Switch
#    parser.add_argument('--opt', action='store_true',
#                        help='A boolean switch')
#
#    # Switch
#    parser.add_argument('--fut', action='store_true',
#                        help='A boolean switch')
#    
#    # Switch
#    parser.add_argument('--db', action='store_true',
#                        help='A boolean switch')
#    
#    
#    args = parser.parse_args()
#    print("Argument values:")
#    print(args.pos_arg)
#    print(args.opt_pos_arg)
#    print(args.domain)
#    print(args.model)
#    print(args.dim)
#    print(args.refit)
#    print(args.post)
#    print(args.opt)
#    print(args.db)
#    
#    if args.pos_arg.upper()  not in valid_pos_args:
#        raise ValueError("%s is not a valid command" % (args.pos_arg))
#        
#    if args.pos_arg.upper() == 'TRAIN':
#        train(args)
#    elif args.pos_arg.upper() == 'FULL_TRAIN':
#        full_train(args)
#    elif args.pos_arg.upper() == 'DB':
#        db_update(args)
#    elif args.pos_arg.upper() == 'PRED':
#        bet_evaluation(args)
#        