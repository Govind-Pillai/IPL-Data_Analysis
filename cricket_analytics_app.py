import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="IPL Analytics Dashboard", layout="wide")

st.title("IPL Analytics Dashboard")
st.markdown("Insights from IPL Matches and Deliveries")

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        matches = pd.read_csv("matches.csv")
        deliveries = pd.read_csv("deliveries.csv")
        
        # Preprocessing
        matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
        matches['year'] = matches['date'].dt.year
        
        merged_data = deliveries.merge(matches, left_on='match_id', right_on='id', how='left')
        return matches, deliveries, merged_data
    except FileNotFoundError:
        st.error("Data files (matches.csv, deliveries.csv) not found. Please ensure they are in the same directory.")
        return None, None, None

matches, deliveries, merged_data = load_data()

if matches is not None:
    # --- Helper: Identify Veterans (2008-2012) ---
    early_seasons = matches[matches['year'].between(2008, 2012)]['id'].unique()
    early_players_bat = deliveries[deliveries['match_id'].isin(early_seasons)]['batter'].unique()
    early_players_bowl = deliveries[deliveries['match_id'].isin(early_seasons)]['bowler'].unique()

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Player Stats", "Stadium Stats", "Team Stats", "Veteran Stats", "Coach Stats"])

    # --- TAB 1: PLAYER STATS ---
    with tab1:
        st.header("Player Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        # FR1: Top Scoring Batsmen Post-2020 (Excl. Veterans)
        with col1:
            st.subheader("Top Scoring Batsmen Post-2020 (Excl. Veterans)")
            post_2020_matches = matches[matches['year'] > 2020]['id'].unique()
            post_2020_data = deliveries[deliveries['match_id'].isin(post_2020_matches)]
            
            fr1_data = post_2020_data[~post_2020_data['batter'].isin(early_players_bat)]
            top_scorers = fr1_data.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)
            
            fig1, ax1 = plt.subplots()
            sns.barplot(x=top_scorers.values, y=top_scorers.index, ax=ax1, palette="viridis")
            ax1.set_xlabel("Total Runs")
            st.pyplot(fig1)

        # FR2: Power Hitters (Strike Rate)
        with col2:
            st.subheader("Top Power Hitters Post-2020 (Strike Rate)")
            stats = post_2020_data.groupby('batter').agg({'batsman_runs': 'sum', 'ball': 'count'})
            stats = stats[stats['ball'] >= 50] # Min 50 balls
            stats['strike_rate'] = (stats['batsman_runs'] / stats['ball']) * 100
            top_sr = stats.sort_values('strike_rate', ascending=False).head(10)
            
            st.dataframe(top_sr[['strike_rate', 'batsman_runs', 'ball']].style.format("{:.2f}"))

        col3, col4 = st.columns(2)

        # FR3: Top Fielders (All Time)
        with col3:
            st.subheader("Top Fielders by Catches (All Time)")
            catches = deliveries[deliveries['dismissal_kind'] == 'caught']
            top_fielders = catches['fielder'].value_counts().head(10)
            
            fig3, ax3 = plt.subplots()
            sns.barplot(x=top_fielders.values, y=top_fielders.index, ax=ax3, palette="magma")
            ax3.set_xlabel("Catches")
            st.pyplot(fig3)

        # FR4: Top Bowlers Post-2020 (Excl. Veterans)
        with col4:
            st.subheader("Top Bowlers Post-2020 (Excl. Veterans)")
            fr4_data = post_2020_data[~post_2020_data['bowler'].isin(early_players_bowl)]
            wickets = fr4_data[fr4_data['is_wicket'] == 1]
            wickets = wickets[~wickets['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])]
            top_bowlers = wickets['bowler'].value_counts().head(10)
            
            fig4, ax4 = plt.subplots()
            sns.barplot(x=top_bowlers.values, y=top_bowlers.index, ax=ax4, palette="coolwarm")
            ax4.set_xlabel("Wickets")
            st.pyplot(fig4)

    # --- TAB 2: STADIUM STATS ---
    with tab2:
        st.header("Stadium & Match Outcome Analysis")

        # Helper functions
        def bat_first_won(row):
            if row['result'] in ['tie', 'no result']: return False
            if row['toss_decision'] == 'bat': return row['toss_winner'] == row['winner']
            else: return row['toss_winner'] != row['winner']

        def field_first_won(row):
            if row['result'] in ['tie', 'no result']: return False
            if row['toss_decision'] == 'field': return row['toss_winner'] == row['winner']
            else: return row['toss_winner'] != row['winner']

        matches['bat_first_win'] = matches.apply(bat_first_won, axis=1)
        matches['field_first_win'] = matches.apply(field_first_won, axis=1)

        # FR5 & FR7: Win Probabilities
        st.subheader("Win Probability by Stadium (Min 10 Matches)")
        
        venue_stats = matches.groupby('venue').agg(
            total_matches=('id', 'count'),
            bat_first_wins=('bat_first_win', 'sum'),
            field_first_wins=('field_first_win', 'sum')
        )
        venue_stats = venue_stats[venue_stats['total_matches'] >= 10]
        venue_stats['Bat 1st Win %'] = (venue_stats['bat_first_wins'] / venue_stats['total_matches']) * 100
        venue_stats['Field 1st Win %'] = (venue_stats['field_first_wins'] / venue_stats['total_matches']) * 100
        
        st.dataframe(venue_stats[['total_matches', 'Bat 1st Win %', 'Field 1st Win %']].style.format("{:.1f}"))

        col5, col6 = st.columns(2)

        # FR6: Toss Impact
        with col5:
            st.subheader("Toss Impact: Win Toss -> Win Match %")
            matches['toss_win_match_win'] = matches['toss_winner'] == matches['winner']
            toss_impact = matches.groupby('venue').agg(
                matches=('id', 'count'),
                wins=('toss_win_match_win', 'sum')
            )
            toss_impact = toss_impact[toss_impact['matches'] >= 10]
            toss_impact['win_prob'] = (toss_impact['wins'] / toss_impact['matches']) * 100
            top_toss_venues = toss_impact.sort_values('win_prob', ascending=False).head(10)
            
            fig6, ax6 = plt.subplots()
            sns.barplot(x=top_toss_venues['win_prob'], y=top_toss_venues.index, ax=ax6, palette="Blues_d")
            ax6.set_xlabel("Win Probability %")
            st.pyplot(fig6)

        # FR8: Avg Runs
        with col6:
            st.subheader("Avg Total Runs: Bat 1st vs Bat 2nd")
            # Simple approximation: Sum of runs per inning / number of innings
            # A more robust way is grouping by match_id and inning
            inning_scores = merged_data.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
            avg_scores = inning_scores.groupby('inning')['total_runs'].mean()
            
            # Filter for inning 1 and 2 only
            avg_scores = avg_scores[avg_scores.index.isin([1, 2])]
            
            fig8, ax8 = plt.subplots()
            sns.barplot(x=avg_scores.index, y=avg_scores.values, ax=ax8)
            ax8.set_xticklabels(['1st Innings', '2nd Innings'])
            ax8.set_ylabel("Avg Runs")
            st.pyplot(fig8)

    # --- TAB 3: TEAM STATS ---
    with tab3:
        st.header("Team Performance Analysis")
        
        col7, col8 = st.columns(2)

        # FR9: Winning % Post-2020
        with col7:
            st.subheader("Team Winning % Post-2020")
            recent_matches = matches[matches['year'] > 2020]
            team_stats = pd.concat([recent_matches['team1'], recent_matches['team2']]).value_counts().reset_index()
            team_stats.columns = ['team', 'matches_played']
            
            wins = recent_matches['winner'].value_counts().reset_index()
            wins.columns = ['team', 'wins']
            
            team_perf = team_stats.merge(wins, on='team', how='left').fillna(0)
            team_perf['win_pct'] = (team_perf['wins'] / team_perf['matches_played']) * 100
            team_perf = team_perf.sort_values('win_pct', ascending=False)
            
            fig9, ax9 = plt.subplots()
            sns.barplot(x=team_perf['win_pct'], y=team_perf['team'], ax=ax9, palette="RdYlGn")
            ax9.set_xlabel("Win %")
            st.pyplot(fig9)

        # FR10: Highest Successful Run Chases
        with col8:
            st.subheader("Highest Successful Run Chases (All Time)")
            # Filter matches where team batting second won
            chase_wins = matches[matches.apply(field_first_won, axis=1)]
            
            # Get target scores (Inning 1 total)
            # We need to sum runs for inning 1 for these match_ids
            inn1_runs = merged_data[merged_data['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
            
            chases = chase_wins.merge(inn1_runs, left_on='id', right_on='match_id')
            top_chases = chases.sort_values('total_runs', ascending=False).head(10)
            
            st.dataframe(top_chases[['season', 'winner', 'venue', 'total_runs']].rename(columns={'total_runs': 'Target Chased'}))

    # --- TAB 4: VETERAN STATS ---
    with tab4:
        st.header("Veteran Player Analysis (Post-2020)")
        st.markdown("Analysis of players active in 2008-2012 who are still playing post-2020.")

        col9, col10 = st.columns(2)

        # FR1 (Veterans): Top Scoring Veterans
        with col9:
            st.subheader("Top Scoring Veterans Post-2020")
            # Filter for veterans
            fr1_vet_data = post_2020_data[post_2020_data['batter'].isin(early_players_bat)]
            top_vet_scorers = fr1_vet_data.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)
            
            fig_v1, ax_v1 = plt.subplots()
            sns.barplot(x=top_vet_scorers.values, y=top_vet_scorers.index, ax=ax_v1, palette="viridis")
            ax_v1.set_xlabel("Total Runs")
            st.pyplot(fig_v1)

        # FR4 (Veterans): Top Bowling Veterans
        with col10:
            st.subheader("Top Bowling Veterans Post-2020")
            fr4_vet_data = post_2020_data[post_2020_data['bowler'].isin(early_players_bowl)]
            vet_wickets = fr4_vet_data[fr4_vet_data['is_wicket'] == 1]
            vet_wickets = vet_wickets[~vet_wickets['dismissal_kind'].isin(['run out', 'retired hurt', 'obstructing the field'])]
            top_vet_bowlers = vet_wickets['bowler'].value_counts().head(10)
            
            fig_v4, ax_v4 = plt.subplots()
            sns.barplot(x=top_vet_bowlers.values, y=top_vet_bowlers.index, ax=ax_v4, palette="coolwarm")
            ax_v4.set_xlabel("Wickets")
            st.pyplot(fig_v4)

    # --- TAB 5: COACH STATS ---
    with tab5:
        st.header("Coach Potential Analysis (Veterans)")
        st.markdown("Evaluating veterans based on team success and experience.")

        # Identify all veterans (batters and bowlers)
        all_veterans = set(early_players_bat) | set(early_players_bowl)
        
        # We need to link players to matches to calculate win %
        # Create a mapping of match_id -> list of players in that match
        # This can be slow, so let's optimize.
        # We can get player-match pairs from deliveries
        player_match = deliveries[['match_id', 'batter', 'bowler', 'non_striker']].melt(id_vars='match_id', value_name='player').drop(columns='variable').drop_duplicates()
        
        # Filter for veterans only to reduce size
        vet_matches = player_match[player_match['player'].isin(all_veterans)]
        
        # Merge with match results
        vet_match_results = vet_matches.merge(matches[['id', 'winner', 'team1', 'team2', 'season', 'toss_decision', 'toss_winner', 'result']], left_on='match_id', right_on='id')
        
        # Determine if the player's team won
        # We don't explicitly know which team a player played for in 'deliveries' easily without more processing
        # BUT, we can infer it:
        # In deliveries, batting_team is available.
        
        # Let's re-do the player-match mapping with team info
        # Get batters and their teams
        batters = deliveries[['match_id', 'batter', 'batting_team']].drop_duplicates().rename(columns={'batter': 'player', 'batting_team': 'team'})
        # Get bowlers and their teams (bowling team is the opposite of batting team, which is tricky)
        # Easier: Just use batters for now, as most veterans bat. Or use the fact that 'bowling_team' exists in some datasets, but here we only have 'batting_team'.
        # Actually, every player bats or bowls.
        # If a player bowled, their team is NOT the batting_team of that row.
        # Let's stick to batters for team mapping to be safe, or assume everyone batted at least once or was listed? No.
        # Let's assume the dataset has 'batting_team' and 'bowling_team' columns?
        # Let's check columns of deliveries.
        # If not, we can infer bowling team.
        
        # Simplification: Use 'batting_team' for batters. For bowlers, we can try to find when they batted.
        # If a player never batted in a match, we might miss their team.
        # However, for "Coach Stats", we are looking at influential players who likely batted.
        
        player_teams = batters
        
        # Filter for veterans
        vet_teams = player_teams[player_teams['player'].isin(all_veterans)]
        
        # Merge with match winner
        vet_perf = vet_teams.merge(matches[['id', 'winner', 'bat_first_win', 'field_first_win']], left_on='match_id', right_on='id')
        
        vet_perf['won'] = vet_perf['team'] == vet_perf['winner']
        
        # FR9 (Coach): Win %
        st.subheader("FR9: Veteran Win % (Min 50 Matches)")
        vet_win_stats = vet_perf.groupby('player').agg(
            matches=('id', 'count'),
            wins=('won', 'sum')
        )
        vet_win_stats = vet_win_stats[vet_win_stats['matches'] >= 50]
        vet_win_stats['win_pct'] = (vet_win_stats['wins'] / vet_win_stats['matches']) * 100
        top_coaches_win = vet_win_stats.sort_values('win_pct', ascending=False).head(10)
        
        st.dataframe(top_coaches_win.style.format({'win_pct': '{:.2f}%'}))

        col11, col12 = st.columns(2)

        # FR10 (Coach): Successful Run Chases
        with col11:
            st.subheader("FR10: Successful Run Chases (Part of Winning Team)")
            # Filter for matches where team fielded first and won (Chase win)
            # Wait, field_first_win means the team that fielded first won.
            # If a player's team won AND the match was won by the team fielding first, then it's a chase win?
            # No, we need to know if the player's team was the one fielding first.
            # If player's team == winner AND field_first_win is True (meaning toss winner chose field and won OR toss winner chose bat and lost... wait)
            # Let's use the 'won' column. If won=True, we just need to know if it was a chase.
            # A chase win happens if the winning team batted second.
            
            # We need to know if 'team' batted second.
            # In 'matches', we have toss_decision and toss_winner.
            # If toss_decision='field' and toss_winner=team -> Batted 2nd.
            # If toss_decision='bat' and toss_winner!=team -> Batted 2nd.
            
            def is_chase_win(row):
                if not row['won']: return False
                # Row has 'team', 'toss_winner', 'toss_decision' (need to merge these)
                return False # Placeholder
            
            # Let's merge toss info back
            vet_perf_full = vet_perf.merge(matches[['id', 'toss_winner', 'toss_decision']], on='id')
            
            def check_chase(row):
                if not row['won']: return False
                if row['toss_decision'] == 'field' and row['toss_winner'] == row['team']: return True
                if row['toss_decision'] == 'bat' and row['toss_winner'] != row['team']: return True
                return False

            vet_perf_full['chase_win'] = vet_perf_full.apply(check_chase, axis=1)
            
            chase_stats = vet_perf_full.groupby('player')['chase_win'].sum().sort_values(ascending=False).head(10)
            
            st.dataframe(chase_stats.to_frame(name="Successful Chases"))

        # FR11 (Coach): Experience
        with col12:
            st.subheader("FR11: Experience (Total Matches Played)")
            # We already calculated matches in vet_win_stats, but that had a filter.
            # Let's recalculate for all veterans
            experience = vet_perf.groupby('player')['id'].count().sort_values(ascending=False).head(10)
            
            fig_c11, ax_c11 = plt.subplots()
            sns.barplot(x=experience.values, y=experience.index, ax=ax_c11, palette="copper")
            ax_c11.set_xlabel("Matches Played")
            st.pyplot(fig_c11)


