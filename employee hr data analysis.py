import pandas as pd

df=pd.read_csv(r'C:\Users\User\Downloads\employee_hr_uncleaned.csv')
df.head()
df.describe()
df.isnull().sum()
df.duplicated().sum()

df.info()

df['age']=df['age'].fillna(0)
df['age']=pd.to_numeric(df['age']).astype(int)
df['years_experience']=df['years_experience'].fillna(0)
df['years_experience']=pd.to_numeric(df['years_experience']).astype(int)
df['annual_salary_inr']=df['annual_salary_inr'].fillna(0)
df['annual_salary_inr']=pd.to_numeric(df['annual_salary_inr']).astype(int)
df['job_satisfaction']=df['job_satisfaction'].fillna(0)
df['job_satisfaction']=pd.to_numeric(df['job_satisfaction']).astype(int)
df.info()

df['employee_id']=df['employee_id'].fillna('unknown')
df['age']=df['age'].fillna(0)
df['gender']=df['age'].fillna('unknown')
df['city']=df['city'].fillna('unknown')
df['department']=df['department'].fillna('unknown')
df['job_role']=df['job_role'].fillna('unknown')
df['education']=df['education'].fillna('unknown')
df['years_experience']=df['years_experience'].fillna(df['years_experience'].mean())
df['annual_salary_inr']=df['annual_salary_inr'].fillna(df['annual_salary_inr'].mean())
df['employment_type']=df['employment_type'].fillna('unknown')
df['performance_rating']=df['performance_rating'].fillna('unknown')
df['overtime_hours_month']=df['overtime_hours_month'].fillna(df['overtime_hours_month'].mean())
df['job_satisfaction']=df['job_satisfaction'].fillna(df['job_satisfaction'].mean())
df['attrition']=df['attrition'].fillna('unknown')
df['joining_date']=df['joining_date'].fillna('unknown')


total_records=df.shape[0]
print('total_records:', total_records)

annual_salary_analysis=df['annual_salary_inr'].mean()
print('annual_salary_analysis:', annual_salary_analysis)

year_experience_analysis=df['years_experience'].mean()
print('year_experience_analysis:', year_experience_analysis)

overtime_hours_month_analysis=df['overtime_hours_month'].mean()
print('overtime_hours_month_analysis:', overtime_hours_month_analysis)

job_satisfaction_analysis=df['job_satisfaction'].mean()
print('job_satisfaction_analysis:', job_satisfaction_analysis)

result=df.columns.to_list()
print(result)

top_rated=df.sort_values(by='years_experience', ascending=False).head(10)
print(top_rated[['employee_id', 'age', 'gender', 'city', 'department', 'job_role', 'education', 'annual_salary_inr', 'employment_type', 'performance_rating', 'overtime_hours_month', 'job_satisfaction', 'attrition', 'joining_date']])

lower_rated=df.sort_values(by='years_experience', ascending=True).head(10)
print(lower_rated[['employee_id', 'age', 'gender', 'city', 'department', 'job_role', 'education','annual_salary_inr', 'employment_type', 'performance_rating', 'overtime_hours_month', 'job_satisfaction', 'attrition', 'joining_date']])

import matplotlib.pyplot as plt
df['annual_salary_inr'].plot(x='annual_salary_inr', y='years_experience', kind='hist')
plt.title('annual salary analysis by year experience')
plt.xlabel('annual_salary_inr')
plt.ylabel('years_experience')
plt.show()

top_rated.plot(kind='bar')
plt.show()

lower_rated.plot(kind='bar')
plt.show()

salary=df.dropna(subset=['annual_salary_inr'])

average=salary.groupby('job_role')['annual_salary_inr'].mean()

job_role=average.idxmax()
annual_salary_analysis=average.max()

print('job role analysis', job_role)
print('annual salary analysis', annual_salary_analysis)

job=df.dropna(subset=['education','job_role'])

average=salary.groupby('job_role')['education'].value_counts()

education=average.index.get_level_values('education').to_list
job_role=average.index.get_level_values('job_role').to_list

print('education', education)
print('job_role', job_role)
print(average.to_string())

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

x=df[['years_experience','annual_salary_inr']]
y=df['job_satisfaction']

scaler=StandardScaler()

x_scaled=scaler.fit_transform(x)

model=LogisticRegression(max_iter=1000)
model.fit(x_scaled,y)

years_experience=int(input('enter year experience:'))
annual_salary_inr=int(input('enter the annual salary:'))

new_data=pd.DataFrame(
   [[years_experience,
annual_salary_inr]],
    columns=['years_experience','annual_salary_inr']
)

new_data_scaled=scaler.fit_transform(new_data)

prediction=model.predict(new_data_scaled)

print('prediction', prediction)

print('classes:', model.classes_)
print('probabilities:', model.predict_proba(new_data_scaled))



