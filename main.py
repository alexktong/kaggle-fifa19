# Maximise overall rating of team constrained by wage budget and playing style

import numpy as np
import pulp as p
import pandas as pd


class OptimalTeam:

    # create position category
    @staticmethod
    def position_category(row):

        if row == 'GK':
            position = 'goalkeeper'
        elif row in ['RB', 'RWB']:
            position = 'right_back'
        elif row in ['LB', 'LWB']:
            position = 'left_back'
        elif 'CB' in row:
            position = 'center_back'
        elif row in ['RM', 'RW']:
            position = 'right_wing'
        elif row in ['LM', 'LW']:
            position = 'left_wing'
        elif any([pos in row for pos in ['AM', 'CM', 'DM']]):
            position = 'center_mid'
        else:
            position = 'forward'

        return position

    # denotes players' market value in dollar term
    @staticmethod
    def value_in_dollar(value):
        return float(value[1:-1]) * 1000

    def __init__(self):

        columns = ['Name', 'Age', 'Club', 'Wage', 'Overall', 'Potential', 'Position']

        self.df = pd.read_csv('archive.zip', usecols=columns)

        # remove rows with missing values
        self.df = self.df.dropna()

        # lower case column names and replace space with underscore
        self.df.columns = [column.lower().replace(' ', '_') for column in self.df.columns]

        # denotes wage in dollar term
        self.df['wage_dollar'] = self.df['wage'].apply(lambda row: self.value_in_dollar(row))

        # create position category
        self.df['position_category'] = self.df['position'].apply(lambda row: self.position_category(row))

    def maximise_overall(self, max_wage):

        df_copy = self.df.copy()

        # create dictionary with players' name as key and overall rating as key value
        overall = dict(zip(df_copy['name'], df_copy['overall']))

        # create dictionary with players' name as key and wage as key value
        wage = dict(zip(df_copy['name'], df_copy['wage_dollar']))

        # create dictionary with players' name as key and position category as key value
        position = dict(zip(df_copy['name'], df_copy['position_category']))

        # create problem variable
        prob = p.LpProblem('overall_team', p.LpMaximize)

        # create decision variable
        names = p.LpVariable.dicts(name='x', indexs=df_copy['name'], lowBound=0, upBound=1, cat='Integer')

        # create objective function
        prob += p.lpSum([overall[i] * names[i] for i in df_copy['name']])

        # create constraint for wage
        prob += p.lpSum([wage[i] * names[i] for i in df_copy['name']]) <= max_wage

        # create constraint for number of goalkeeper(s)
        prob += p.lpSum([names[i] for i in df_copy['name'] if position[i] == 'goalkeeper']) == 3

        # create constraint for number of right back(s)
        prob += p.lpSum([names[i] for i in df_copy['name'] if position[i] == 'right_back']) == 2

        # create constraint for number of left back(s)
        prob += p.lpSum([names[i] for i in df_copy['name'] if position[i] == 'left_back']) == 2

        # create constraint for number of center back(s)
        prob += p.lpSum([names[i] for i in df_copy['name'] if position[i] == 'center_back']) == 5

        # create constraint for number of right wing(s)
        prob += p.lpSum([names[i] for i in df_copy['name'] if position[i] == 'right_wing']) == 2

        # create constraint for number of left wing(s)
        prob += p.lpSum([names[i] for i in df_copy['name'] if position[i] == 'left_wing']) == 3

        # create constraint for number of center midfielders(s)
        prob += p.lpSum([names[i] for i in df_copy['name'] if position[i] == 'center_mid']) == 7

        # create constraint for number of forward(s)
        prob += p.lpSum([names[i] for i in df_copy['name'] if position[i] == 'forward']) == 4

        # create dictionary with players' name as key and club as key value
        club = dict(zip(df_copy['name'], df_copy['club']))

        # create constraint for number of players in same club
        for club_name in df_copy['club'].unique():
            prob += p.lpSum([names[i] for i in df_copy['name'] if club[i] == club_name]) <= 2

        # run solver
        prob.solve()

        return prob
