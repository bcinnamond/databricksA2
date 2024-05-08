# Databricks notebook source
# DBTITLE 1,Importing functions
from pyspark.sql.functions import col, sum, rank, date_format, monotonically_increasing_id, round, asc, desc, countDistinct
from pyspark.sql.types import IntegerType, DateType, DoubleType
import pandas as pd
import plotly.express as px


# COMMAND ----------

# DBTITLE 1,Mount the Container
dbutils.fs.mount(
    source='wasbs://project-games-data@rcinnamondsa.blob.core.windows.net',
    mount_point='/mnt/project-games-data',
    extra_configs = {'fs.azure.account.key.rcinnamondsa.blob.core.windows.net': dbutils.secrets.get('GamesSecretScope', 'storageAccountKey')}
)


# COMMAND ----------

# DBTITLE 1,List the content of the container
# MAGIC %fs 
# MAGIC ls "/mnt/project-games-data"

# COMMAND ----------

# DBTITLE 1,Load games CSV into a dataframe taking first row as header
games = spark.read.format("csv").option("header","true").load("/mnt/project-games-data/raw-data/vgsales.csv").createOrReplaceTempView("temp_table")

# COMMAND ----------

# DBTITLE 1,Create Spark table from the temporary view
spark.sql("CREATE TABLE IF NOT EXISTS gamestb USING parquet AS SELECT * FROM temp_table")

# COMMAND ----------

# DBTITLE 1,Load Spark table into a data frame
gamestb = spark.table("gamestb")

# COMMAND ----------

# DBTITLE 1,Print the dataframe schema
gamestb.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC After reviewing the schema it is clear that some data types should be changed. The Rank column should be an integer data type to allow correct sorting, Year should be a date type and the Sales fields should be doubles to allow for calculations. 

# COMMAND ----------

# DBTITLE 1,Cast the rank column as an integer

gamestb = gamestb.withColumn("Rank", col("Rank").cast(IntegerType()))

# COMMAND ----------

# DBTITLE 1,Cast the year column as a date
gamestb = gamestb.withColumn("Year", col("Year").cast(DateType()))

# COMMAND ----------

# DBTITLE 1,Cast the sales columns as doubles
gamestb = gamestb.withColumn("NA_Sales", col("NA_Sales").cast(DoubleType()))
gamestb = gamestb.withColumn("EU_Sales", col("EU_Sales").cast(DoubleType()))
gamestb = gamestb.withColumn("JP_Sales", col("JP_Sales").cast(DoubleType()))
gamestb = gamestb.withColumn("Other_Sales", col("Other_Sales").cast(DoubleType()))
gamestb = gamestb.withColumn("Global_Sales", col("Global_Sales").cast(DoubleType()))

# COMMAND ----------

# DBTITLE 1,Print dataframe schema to view changes
gamestb.printSchema()

# COMMAND ----------

# DBTITLE 1,All time top 10 sales
top_10_highest_sales = gamestb.orderBy("Rank").limit(10).show()

# COMMAND ----------

# DBTITLE 1,All time top 10 action games
top_10_highest_sales = gamestb.orderBy("Rank").filter(col("Genre").contains("Action")).limit(10).show()

# COMMAND ----------

# MAGIC %md
# MAGIC After these queries I could see that I would require extra totals columns to group sales and aggregate the totals fields. I also thought there might be some discrepancies in the Global_Sales values so I recalculated the values in that column. As the Rank column relies on the Global_Sales values I renumbered it using an autonumber function.

# COMMAND ----------

# DBTITLE 1,After finding null values further down this step was added to ensure below data is valid
gamestb.dropna()

# COMMAND ----------

# DBTITLE 1,Format the year columns datatype as Year
gamestb = gamestb.withColumn("year", date_format(col("year"), "yyyy"))

# COMMAND ----------

# DBTITLE 1,Validate the Global sales values
gamestb = gamestb.withColumn("Global_Sales", col("NA_Sales") + col("EU_Sales") + col("JP_Sales") + col("Other_Sales") )

# COMMAND ----------

# DBTITLE 1,Create a new dataframe with rank sorted desc
sorted_gamestb = gamestb.orderBy(col("Global_Sales").desc())

# COMMAND ----------

# DBTITLE 1,Repopulate rank based on Global_Sales and row pointer
gamestb = sorted_gamestb.withColumn("Rank", monotonically_increasing_id() + 1)

# COMMAND ----------

# DBTITLE 1,Create dataframe to aggregate sales by name
total_sales_df = gamestb.groupBy("Name").agg(
    sum("NA_Sales").alias("Total_NA_Sales"),
    sum("EU_Sales").alias("Total_EU_Sales"),
    sum("JP_Sales").alias("Total_JP_Sales"),
    sum("Other_Sales").alias("Total_Other_Sales"),
    sum("Global_Sales").alias("Total_Global_Sales")
)

# COMMAND ----------

# DBTITLE 1,Calculate total sales per game
total_sales_df = total_sales_df.withColumn("Game_Total_Sales", 
                                           total_sales_df["Total_NA_Sales"] + 
                                           total_sales_df["Total_EU_Sales"] + 
                                           total_sales_df["Total_JP_Sales"] + 
                                           total_sales_df["Total_Other_Sales"])

# COMMAND ----------

# DBTITLE 1,Join new column as new dataframe
gamestb_with_totals = gamestb.join(total_sales_df.select("Name", "Game_Total_Sales"), "Name", "left")

# COMMAND ----------

# DBTITLE 1,Create dataframe to aggregate sales by publisher
publisher_sales_df = gamestb.groupBy("Publisher").agg(
    sum("NA_Sales").alias("Total_NA_Sales"),
    sum("EU_Sales").alias("Total_EU_Sales"),
    sum("JP_Sales").alias("Total_JP_Sales"),
    sum("Other_Sales").alias("Total_Other_Sales"),
    sum("Global_Sales").alias("Total_Global_Sales")
)

# COMMAND ----------

# DBTITLE 1,Calculate total sales by publisher
publisher_sales_df = publisher_sales_df.withColumn("Total_Sales", 
                                                   publisher_sales_df["Total_NA_Sales"] + 
                                                   publisher_sales_df["Total_EU_Sales"] + 
                                                   publisher_sales_df["Total_JP_Sales"] + 
                                                   publisher_sales_df["Total_Other_Sales"])

# COMMAND ----------

# DBTITLE 1,Join new column as new dataframe
df_with_totals_and_publisher_sales = gamestb_with_totals.join(
    publisher_sales_df.select("Publisher", "Total_Sales").withColumnRenamed("Total_Sales", "Publisher_Total_Sales"), 
    "Publisher", "left")

# COMMAND ----------

# DBTITLE 1,Create a new dataframe to aggregate sales by platform
platform_sales_df = gamestb.groupBy("Platform").agg(
    sum("NA_Sales").alias("Total_NA_Sales"),
    sum("EU_Sales").alias("Total_EU_Sales"),
    sum("JP_Sales").alias("Total_JP_Sales"),
    sum("Other_Sales").alias("Total_Other_Sales"),
    sum("Global_Sales").alias("Total_Global_Sales")
)

# COMMAND ----------

# DBTITLE 1,Calculate total sales by publisher
platform_sales_df = platform_sales_df.withColumn("Platform_Total_Sales", 
                                                 platform_sales_df["Total_NA_Sales"] + 
                                                 platform_sales_df["Total_EU_Sales"] + 
                                                 platform_sales_df["Total_JP_Sales"] + 
                                                 platform_sales_df["Total_Other_Sales"])

# COMMAND ----------

# DBTITLE 1,Join new column to dataframe
df_with_totals_publisher_platform = df_with_totals_and_publisher_sales.join(
    platform_sales_df.select("Platform", "Platform_Total_Sales"), 
    "Platform", "left"
)

# COMMAND ----------

# DBTITLE 1,Create a new dataframe to aggregate sales by Genre
genre_sales_df = gamestb.groupBy("Genre").agg(
    sum("NA_Sales").alias("Total_NA_Sales"),
    sum("EU_Sales").alias("Total_EU_Sales"),
    sum("JP_Sales").alias("Total_JP_Sales"),
    sum("Other_Sales").alias("Total_Other_Sales"),
    sum("Global_Sales").alias("Total_Global_Sales")
)

# COMMAND ----------

# DBTITLE 1,Calculate total sales by genre
genre_sales_df = genre_sales_df.withColumn("Genre_Total_Sales", 
                                                 genre_sales_df["Total_NA_Sales"] + 
                                                 genre_sales_df["Total_EU_Sales"] + 
                                                 genre_sales_df["Total_JP_Sales"] + 
                                                 genre_sales_df["Total_Other_Sales"])

# COMMAND ----------

# DBTITLE 1,Join new column to dataframe
df_with_totals_publisher_platform_genre = df_with_totals_publisher_platform.join(
    genre_sales_df.select("Genre", "Genre_Total_Sales"), 
    "Genre", "left"
)

# COMMAND ----------

# MAGIC %md
# MAGIC After creating all my new totals columns I print the schema and sort into their original order. I also copy the data into the transformed data folder in my storage account container.

# COMMAND ----------

# DBTITLE 1,Display dataframe with new totals
df_with_totals_publisher_platform_genre.show()

# COMMAND ----------

df_with_totals_publisher_platform_genre.printSchema()

# COMMAND ----------

# DBTITLE 1,Reorder the dataframe
finalgames = df_with_totals_publisher_platform_genre.select(
    col("Rank"),
    col("Name"),
    col("Platform"),
    col("year").cast("date").alias("Year"),
    col("Genre"),
    col("Publisher"),
    col("NA_Sales"),
    col("EU_Sales"),
    col("JP_Sales"),
    col("Other_Sales"),
    col("Global_Sales"),
    col("Game_Total_Sales"),
    col("Publisher_Total_Sales"),
    col("Platform_Total_Sales"), 
    col("Genre_Total_Sales"), 
)

# COMMAND ----------

# DBTITLE 1,Copy the dataframe into transformed-data
finalgames.write.option("header", 'true').mode("overwrite").csv("/mnt/project-games-data/transformed-data/games")

# COMMAND ----------

finalgames.printSchema()

# COMMAND ----------

finalgames.show()

# COMMAND ----------

# DBTITLE 1,Game Popularity
all_time_top_10 = finalgames.select("Name", col("Game_Total_Sales").alias("Total Sales")).distinct().orderBy(finalgames["Game_Total_Sales"].desc()).limit(10)

all_time_top_10.show()

# COMMAND ----------

# DBTITLE 1,Game Popularity Visualisation with Plotly
all_time_top_10_pd = all_time_top_10.toPandas()

fig = px.bar(all_time_top_10_pd, x='Name', y='Total Sales', title='Top 10 Games by Total Sales (in Millions)', color='Name')
fig.show()

# COMMAND ----------

# DBTITLE 1,Genre Popularity
all_time_genre = finalgames.select("Genre", col("Genre_Total_Sales").alias("Total Sales")).distinct().orderBy(finalgames["Genre_Total_Sales"].desc())

all_time_genre.show()

# COMMAND ----------

# DBTITLE 1,Genre Popularity Visualisation with Plotly
all_time_genre_pd = all_time_genre.toPandas()

fig = px.bar(all_time_genre_pd, x='Genre', y='Total Sales', title='Top 10 Genres by Total Sales (in Millions)', color='Genre')
fig.show()

# COMMAND ----------

# DBTITLE 1,Platform Popularity
all_time_top_10_platform = finalgames.select("Platform", "Platform_Total_Sales").distinct().orderBy(finalgames["Platform_Total_Sales"].desc()).limit(10)

all_time_top_10_platform.show()

# COMMAND ----------

# DBTITLE 1,Platform Popularity Visualisation with Plotly
all_time_top_10_platform_pd = all_time_top_10_platform.toPandas()

fig = px.pie(all_time_top_10_platform_pd, values='Platform_Total_Sales', names='Platform', title='Top 10 Platforms by Total Sales (in Millions)')
fig.show()

# COMMAND ----------

# DBTITLE 1,Publisher Popularity
all_time_top_10_publisher = finalgames.select("Publisher", "Publisher_Total_Sales").distinct().orderBy(finalgames["Publisher_Total_Sales"].desc()).limit(10)

all_time_top_10_publisher.show()

# COMMAND ----------

# DBTITLE 1,Publisher Popularity Visualisation with Plotly
all_time_top_10_publisher_pd = all_time_top_10_publisher.toPandas()

fig = px.pie(all_time_top_10_publisher_pd, values='Publisher_Total_Sales', names='Publisher', title='Top 10 Publishers by Total Sales (in Millions)')
fig.show()

# COMMAND ----------

# DBTITLE 1,Top Ranked Games
top_ranked_games = finalgames.select("Rank", "Name", "Platform", "Global_Sales", "Year").orderBy(finalgames["Rank"].asc()).limit(10)

top_ranked_games.show()

# COMMAND ----------

# DBTITLE 1,Top Ranked Games Visualisation with Plotly
top_ranked_games_pd = top_ranked_games.toPandas()

platform_colors = {
    'Wii': 'blue',
    'NES': 'green',
    'Gameboy': 'red',
    'DS': 'purple',
    'Mobile': 'orange',
    'Switch': 'yellow'
}

top_ranked_games_pd['Color'] = top_ranked_games_pd['Platform'].map(platform_colors)

fig = px.bar(top_ranked_games_pd.sort_values(by='Rank'), x='Name', y='Rank', title='Top 10 Ranked Games', color='Platform', color_discrete_map=platform_colors)

fig.show()

# COMMAND ----------

# DBTITLE 1,Top 10 Ranked Games since 2010
top_ranked_games_2010 = finalgames.select("Rank", "Name", "Platform", "Global_Sales", "Year").filter(col("Year") >= "2010-01-01").orderBy(finalgames["Rank"].asc()).limit(10)

top_ranked_games_2010.show()

# COMMAND ----------

# DBTITLE 1,Top 10 Ranked since 2010 Visualisation with Plotly
top_ranked_games_2010_pd = top_ranked_games_2010.toPandas()

platform_colors = {
    'PS3': 'blue',
    'X360': 'green',
    'PS4': 'red',
    'DS': 'purple',
    '3DS': 'orange'
}

top_ranked_games_2010_pd['Color'] = top_ranked_games_pd['Platform'].map(platform_colors)

fig = px.bar(top_ranked_games_2010_pd.sort_values(by='Rank'), x='Name', y='Rank', title='Top 10 Ranked Games', color='Platform', color_discrete_map=platform_colors)

fig.show()

# COMMAND ----------

# DBTITLE 1,Top 10 PS4 Games
top_10_ps4 = finalgames.select("Name", "Platform", "Global_Sales").filter(col("Platform")==("PS4")).orderBy(finalgames["Global_Sales"].desc()).limit(10)

top_10_ps4.show()

# COMMAND ----------

# DBTITLE 1,Top 10 PS4 Games Visualisation with Plotly
top_10_ps4_pd = top_10_ps4.toPandas()

fig = px.pie(top_10_ps4_pd, values='Global_Sales', names='Name', title='Top 10 PS4 Games by Total Sales (in Millions)')
fig.show()

# COMMAND ----------

# DBTITLE 1,Top 10 Games in 2001
top_10_2001 = finalgames.select("Name", "Year", "Game_Total_Sales").filter(col("Year").contains("2001")).distinct().orderBy(finalgames["Game_Total_Sales"].desc()).limit(10)

top_10_2001.show()

# COMMAND ----------

top_10_2001_pd = top_10_2001.toPandas()

fig = px.pie(top_10_2001_pd, values='Game_Total_Sales', names='Name', title='Top 10 Games in 2001 by Total Sales (in Millions)')
fig.show()

# COMMAND ----------

# DBTITLE 1,Top 10 Action Games in 2010
action_games_2010 = finalgames.filter((col("Year") == "2010") & (col("Genre") == "Action"))

action_sales_2010 = action_games_2010.groupBy("Name").agg(sum("Global_Sales").alias("Total Sales"))

top_10_action_2010 = action_sales_2010.orderBy(action_sales_2010["Total Sales"].desc()).limit(10)

top_10_action_2010.show()

# COMMAND ----------

# DBTITLE 1,Top 10 Action Games in 2010 Visualisation with Plotly
top_10_action_2010_pd = top_10_action_2010.toPandas()

fig = px.bar(top_10_action_2010_pd, x='Name', y='Total Sales', title='Top 10 Action Games in 2010 by Total Sales (in Millions)',color='Name')
fig.show()

# COMMAND ----------

# DBTITLE 1,Top 10 Games in the EU in 2012
games_2012 = finalgames.filter(finalgames["Year"] == "2012")

eu_sales_by_game = games_2012.groupBy("Name").agg(sum("EU_Sales").alias("Total EU Sales"))

top_10_eu_2012 = eu_sales_by_game.orderBy(eu_sales_by_game["Total EU Sales"].desc()).limit(10)

top_10_eu_2012.show()

# COMMAND ----------

# DBTITLE 1,Top 10 Games in the EU in 2012 Visualisations with Plotly
top_10_eu_2012_pd = top_10_eu_2012.toPandas()

fig = px.bar(top_10_eu_2012_pd, x='Name', y='Total EU Sales', title='Top 10 Games in the EU in 2012 by Total Sales (in Millions)')
fig.show()

# COMMAND ----------

# DBTITLE 1,Activision Performance Over Time

publisher_performance_over_time = finalgames.groupBy("Year", "Publisher") \
    .agg(sum("Global_Sales").alias("Total Sales")).filter(col("Publisher") == "Activision").orderBy(finalgames["Year"].desc())

publisher_performance_over_time.show()

# COMMAND ----------

# DBTITLE 1,Activision Performance Visualisation with Plotly
publisher_performance_over_time_pd = publisher_performance_over_time.toPandas()

fig = px.line(publisher_performance_over_time_pd, x="Year", y="Total Sales", title="Activision Performance")
fig.show()

# COMMAND ----------

# DBTITLE 1,Validation One Year per Game
just_cause_2_df = finalgames.select("Name", "Year").filter(col("Name") == "Just Cause 2")

just_cause_2_df.show()

# COMMAND ----------

# DBTITLE 1,Games Released Per Year
games_released_per_year = finalgames.groupBy("Year") \
    .agg(countDistinct("Name").alias("Games Released")).orderBy(finalgames["Year"].desc())

games_released_per_year.show()


# COMMAND ----------

# DBTITLE 1,Games Released per Year Visualisation with Plotly
games_released_per_year_pd = games_released_per_year.toPandas()

fig = px.line(games_released_per_year_pd, x="Year", y="Games Released", title="Games Released per Year")
fig.show()
