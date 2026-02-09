import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import streamlit as st
import json
import os
import re

st.set_page_config(page_title = 'Finance App', page_icon = '💸', layout='wide')

# reads file, if successful then converts to a dataframe coverting to the proper mm/dd/yyyy format and dropping unneeded columns
# throws error if not successful 
def load_transactions(file):
    try:
        df = pd.read_csv(file)
        df.columns = [col.strip() for col in df.columns]
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
        df.drop(['Post Date', 'Memo'], axis=1, inplace=True)
        return df
    except Exception as e:
        st.error(f'Error processing file: {str(e)}')
        return None


#combines all amazon purchases under the same name
def normalize_descriptions(df):
    MERCHANT_PATTERNS = {
    'AMAZON':      r'\b(amzn|amazon)\b',
    'ALIEXPRESS':  r'\b(aliexpress)\b',
    'YESSTYLE':    r'\b(yesstyle)\b',
    'STYLEVANA':   r'\b(stylevana|stylevane)\b',
    'UNIQLO':      r'\b(uniqlo)\b',
    'H MART':      r'\b(h\s*mart)\b',
    'TARGET':      r'\b(target)\b',
    'TASTEA':      r'\b(tastea)\b',
    'IN N OUT':    r'\b(in[\s-]*n[\s-]*out)\b',
    'RIOT':        r'\b(riot)\b',
    }
    def normalize(desc):
        for merchant, pattern in MERCHANT_PATTERNS.items():
            if re.search(pattern, desc, re.I):
                return merchant
        return desc
    
    df['Description'] = df['Description'].apply(normalize)
    return df

# opens webpage using streamlit 
def main():
    st.title('Finance Dashboard')
    upload_file = st.file_uploader('Upload your transaction csv file', type=['csv'])
    if upload_file is not None:
        df = load_transactions(upload_file)
        df = normalize_descriptions(df)
        spendings_df = df[df['Type'] != 'Payment']
    
        tab1, tab2, tab3= st.tabs(['All Transactions', 'Montly Spendings', 'Spending Predictions'])
        with tab1:
            all_trans = df.copy()
            all_trans['Transaction Date'] = all_trans['Transaction Date'].dt.date
            st.subheader('All Transactions')
            st.write(all_trans)           

            st.subheader('Expense Summary')
            category_totals = all_trans.groupby('Category')['Amount'].sum().abs().reset_index().sort_values('Amount', ascending=False)
            st.dataframe(category_totals,
                         column_config={
                             'Amount' : st.column_config.NumberColumn('Amount', format='%.2f USD')
                         },
                         use_container_width=True,
                         hide_index=True)
            
            category_pie = px.pie(category_totals,
                         values='Amount',
                         names='Category',
                         title='Expenses by Category')
            st.plotly_chart(category_pie, use_container_width=True)
            
            st.subheader('Top 10 Expenses')
            total_by_place = spendings_df.groupby('Description')['Amount'].sum().abs().reset_index().sort_values('Amount', ascending=False).head(10)
            st.dataframe(total_by_place,
                         column_config={
                             'Amount' : st.column_config.NumberColumn('Amount', format='%.2f USD')
                         },
                         use_container_width=True,
                         hide_index=True)
            
            st.subheader('Yearly Spendings')
            yearly_spendings = spendings_df.groupby(df['Transaction Date'].dt.year)['Amount'].sum().abs().reset_index().sort_values('Transaction Date', ascending=True)
            st.dataframe(yearly_spendings,
                         column_config={
                             'Amount' : st.column_config.NumberColumn('Amount', format='%.2f USD')
                         },
                         use_container_width=True,
                         hide_index=True)
            
        with tab2:
            monthly_df = spendings_df.copy()
            monthly_df['Transaction Date'] = monthly_df['Transaction Date'].dt.to_period('M')
            spendings_by_month_year = monthly_df.groupby('Transaction Date')['Amount'].sum().abs().reset_index()
            spendings_by_month_year['Transaction Date'] = (spendings_by_month_year['Transaction Date'].dt.to_timestamp())

            month_options = (
            spendings_by_month_year['Transaction Date']
            .dt.to_period('M')
            .sort_values(ascending=False)
            .astype(str)
            .unique())
            month_options = ['Choose a month', 'Past 6 months', 'Quarterly Summary'] + list(month_options)

            selected_month = st.selectbox('Select a month', month_options)



            month_line_graph = px.line(spendings_by_month_year,
                                       x='Transaction Date',
                                       y='Amount',
                                       title='Spending by All Months')
            
            month_line_graph.update_xaxes(
            dtick='M1',
            tickformat='%b %Y',
            ticklabelmode='period'
            )


#summarize spendings for each month by categories (pie chart) and line graph showing spendings for each day
            if selected_month == 'Choose a month':
                st.plotly_chart(month_line_graph, use_container_width=True)

            else:
                if selected_month == 'Past 6 months':
                    most_recent = spendings_df['Transaction Date'].max()
                    cutoff = most_recent - pd.DateOffset(months=6)
                    filtered_df = spendings_df[spendings_df['Transaction Date'] >= cutoff]
                    #pie chart showing summary by category for the past 6 months
                    spending_by_category = (filtered_df.groupby(['Category', 'Description'])['Amount'].sum().abs().reset_index()
                                            .sort_values(['Category', 'Amount'], ascending=[True, False]))
                    top_3_per_category = spending_by_category.groupby('Category').head(3)

                    category_sum = filtered_df.groupby('Category')['Amount'].sum().abs().reset_index().sort_values('Amount', ascending=False)
                    category_pie = px.pie(category_sum,
                                        values='Amount',
                                        names='Category',
                                        title='Expenses by Category')
                    #bar graph showing spending summary for the past 6 months
                    filtered_month = filtered_df.copy()
                    filtered_month['Transaction Date'] = filtered_month['Transaction Date'].dt.to_period('M')
                    spendings_6months = filtered_month.groupby('Transaction Date')['Amount'].sum().abs().reset_index()
                    spendings_6months['Transaction Date'] = (spendings_6months['Transaction Date'].dt.to_timestamp())
                    
                    month_bar_graph = px.bar(spendings_6months,
                                       x='Transaction Date',
                                       y='Amount',
                                       title='Spending by All Month',
                                       text='Amount')
                    
                    month_bar_graph.update_layout(
                        xaxis_title='Month',
                        yaxis_title='Total Spending (USD)')

                    month_bar_graph.update_traces(
                        textposition='outside',
                        texttemplate='%{text:.2f}')                

                    #displays graphs                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(category_pie, use_container_width=True)
                        st.subheader('Top 3 Expenses per Category')
                        st.dataframe(top_3_per_category,
                                     column_config={
                                         'Amount' : st.column_config.NumberColumn('Amount', format='%.2f USD')},
                                     use_container_width=True,
                                     hide_index=True)

                    with col2:
                        st.plotly_chart(month_bar_graph, use_container_width=True)
                    
                    #Top Expenses
                    st.subheader('Top Expenses by Place')
                    top_spendings = filtered_df.groupby('Description')['Amount'].sum().abs().reset_index().sort_values('Amount', ascending=False).head(5)
                    st.dataframe(top_spendings,
                         column_config={
                             'Amount' : st.column_config.NumberColumn('Amount', format='%.2f USD')
                         },
                         use_container_width=True,
                         hide_index=True)


                elif selected_month == 'Quarterly Summary':
                    quarterly_df = spendings_df.copy()
                    quarterly_df['Year'] = quarterly_df['Transaction Date'].dt.year
                    quarterly_df['Quarter'] = quarterly_df['Transaction Date'].dt.quarter
                    quarterly_df['Transaction Date'] = quarterly_df['Transaction Date'].dt.to_period('Q')
                    quarterly_df['Transaction Date'] = ('Q' + quarterly_df['Transaction Date'].dt.quarter.astype(str)+ ' ' + quarterly_df['Transaction Date'].dt.year.astype(str))
                    Q_sums = quarterly_df.groupby('Transaction Date')['Amount'].sum().abs().reset_index()
                    Q_sums_pie = px.bar(Q_sums,
                                        y='Amount',
                                        x='Transaction Date',
                                        title='Spendings by Quarter',
                                        text='Amount')
                    
                    Q_sums_pie.update_layout(
                        xaxis_title='Quarter',
                        yaxis_title='Total Spending (USD)')

                    Q_sums_pie.update_traces(
                        textposition='outside',
                        texttemplate='%{text:.2f}')
                    
                    st.plotly_chart(Q_sums_pie, use_container_width=True)
                    
                    st.subheader('Top 3 Expenses Each Quarter')
                    top_each_quarter = (quarterly_df.groupby(['Year', 'Quarter', 'Transaction Date', 'Category', 'Description'])['Amount']
                                        .sum().abs().reset_index()
                                        .sort_values(['Year', 'Quarter', 'Amount'], ascending=[True, True, False]))
                    top_each_quarter = top_each_quarter.drop(columns=['Year', 'Quarter']).groupby('Transaction Date').head(3)
                    st.dataframe(top_each_quarter,
                         column_config={
                             'Amount' : st.column_config.NumberColumn('Amount', format='%.2f USD')
                         },
                         use_container_width=True,
                         hide_index=True)
                else: 
                    filter_month_df = spendings_df.copy()
                    filter_month_df['MonthYear'] = filter_month_df['Transaction Date'].dt.to_period('M')
                    filter_month_df = filter_month_df[filter_month_df['MonthYear'] == selected_month]
                    filter_month_df['Amount'] = filter_month_df['Amount'].abs()
                    formatted_df = filter_month_df.copy().drop(columns=['MonthYear']) #copys the filtered by month df and changes datetime to correct format YYYY-MM-DD
                    formatted_df['Transaction Date'] = formatted_df['Transaction Date'].dt.date
                    st.subheader('All Transactions')
                    st.dataframe(formatted_df,
                         column_config={
                             'Amount' : st.column_config.NumberColumn('Amount', format='%.2f USD')
                         },
                         use_container_width=True,
                         hide_index=True)
                    #line graph showing expense throughout the month
                    #axis formatting for each month
                    start = filter_month_df['Transaction Date'].min().replace(day=1)
                    end = (start + pd.offsets.MonthEnd(1))
                    all_days = pd.DataFrame({'Transaction Date': pd.date_range(start, end, freq='D')})
                    daily = all_days.merge(filter_month_df[['Transaction Date', 'Amount']],on='Transaction Date',how='left')
                    daily['Amount'] = daily['Amount'].fillna(0)
                    daily = daily.groupby('Transaction Date')['Amount'].sum().abs().reset_index()
                    
                    spending_per_day = px.line(daily,
                                               x='Transaction Date',
                                               y='Amount',
                                               title='Monthly Spending')
                    
                    spending_per_day.update_xaxes(
                        dtick=24 * 60 * 60 * 1000,
                        tickformat='%b %d')
                    
                    # pie chart showing categorical spendings
                    category_spendings = filter_month_df.groupby('Category')['Amount'].sum().reset_index()
                    category_spendings_chart = px.pie(category_spendings,
                                                      values='Amount',
                                                      names='Category',
                                                      title='Expenses by Category')
                    
                    col1, col2= st.columns(2)
                    
                    with col1: #spendings per day line graph
                        st.plotly_chart(spending_per_day, use_container_width=True)
                    with col2: #categorical spendings
                        st.plotly_chart(category_spendings_chart, use_container_width=True) 
                             
                    #top 3 expenses that month 
                    st.subheader('Top 5 Expenses')
                    top_3_expenses = formatted_df.sort_values('Amount', ascending=False).head(5)
                    st.dataframe(top_3_expenses,
                         column_config={
                             'Amount' : st.column_config.NumberColumn('Amount', format='%.2f USD')
                         },
                         use_container_width=True,
                         hide_index=True)
        with tab3: #spendings predictions for 1 and 3 months
            previous_months = spendings_df.copy()
            previous_months['MonthYear'] = previous_months['Transaction Date'].dt.to_period('M')
            monthly = (
                previous_months.groupby('MonthYear')['Amount']
                .sum()
                .abs()
                .reset_index())

            monthly['MonthYear'] = monthly['MonthYear'].dt.to_timestamp()
            #spendings for the past 1,2,3 months
            monthly['past month'] = monthly['Amount'].shift(1)
            monthly['past 2 month'] = monthly['Amount'].shift(2)
            monthly['past 3 month'] = monthly['Amount'].shift(3)
            #average spendings over the past 3 and 6 months
            monthly['3 month avg'] = monthly['Amount'].rolling(3).mean()
            monthly['6 month avg'] = monthly['Amount'].rolling(6).mean()
            monthly = monthly.dropna()
            
            #Linear Regression
            feature_cols = ['past month', 'past 2 month', 'past 3 month', '3 month avg', '6 month avg']
            X = monthly[feature_cols]
            y = monthly['Amount']
            model = LinearRegression()
            model.fit(X, y)
            
            last_month = monthly.iloc[-1]
            def predict_next(model, last_row): #Predict the next month's spending using the trained model
                features = [[
                    last_row['past month'],
                    last_row['past 2 month'],
                    last_row['past 3 month'],
                    last_row['3 month avg'],
                    last_row['6 month avg']]]
                
                return model.predict(features)[0]
            
            # 1-month prediction
            next_month = predict_next(model, last_month)
            
            # 3-month prediction 
            predicted_spending = []
            temp = last_month.copy()
            for i in range(3):
                pred = predict_next(model, temp)
                predicted_spending.append(pred)
                # update lags
                temp['past 3 month'] = temp['past 3 month']
                temp['past 2 month'] = temp['past 2 month']
                temp['past month'] = pred
                
            st.markdown(f"""
                        # 📊 Spending Predictions
                        ## **Next month:** ${next_month:,.2f}""") 
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                        ## **Next 3 months:**  
                        ### 1. ${predicted_spending[0]:,.2f}  
                        ### 2. ${predicted_spending[1]:,.2f}  
                        ### 3. ${predicted_spending[2]:,.2f}""")
            with col2:
                last_month = pd.to_datetime(monthly['MonthYear'].max())
                future_dates = pd.date_range(
                    start=last_month + pd.offsets.MonthBegin(1),
                    periods=3,
                    freq='MS')

                
                pred_df = pd.DataFrame({
                'Month': future_dates,
                'Predicted Spending': predicted_spending})
                
                pred_df['Month'] = pred_df['Month'].dt.strftime('%b %Y')
                
                pred_bar = px.bar(pred_df,
                                  x='Month',
                                  y='Predicted Spending',
                                  title='Predicted Spending for the Next 3 Months',
                                  text='Predicted Spending')
                pred_bar.update_layout(
                xaxis_title='Month',
                yaxis_title='Predicted Spending (USD)')
            
                pred_bar.update_traces(textposition='outside',
                                       texttemplate='%{text:.2f}') 
                
                pred_bar.update_xaxes(tickformat='%b')
            
                st.plotly_chart(pred_bar, use_container_width=True)


main()
