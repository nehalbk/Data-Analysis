-- Creating the database
CREATE DATABASE IF NOT EXISTS walmartSales;
use walmartSales;

-- preview data
select * from sales limit 10;

-- Creating table
CREATE TABLE IF NOT EXISTS sales(
	invoice_id VARCHAR(30) NOT NULL PRIMARY KEY,
    branch VARCHAR(5) NOT NULL,
    city VARCHAR(30) NOT NULL,
    customer_type VARCHAR(30) NOT NULL,
    gender VARCHAR(30) NOT NULL,
    product_line VARCHAR(100) NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    quantity INT NOT NULL,
    tax_pct FLOAT NOT NULL,
    total DECIMAL(12, 4) NOT NULL,
    date DATE NOT NULL,
    time TIME NOT NULL,
    payment VARCHAR(15) NOT NULL,
    cogs DECIMAL(10,2) NOT NULL,
    gross_margin_pct FLOAT,
    gross_income DECIMAL(12, 4),
    rating FLOAT
);

-- --------------------------------------Feature Engineering--------------------------------------------------------------

-- time of the day

-- test the column
SELECT time,
	(
		CASE
			WHEN `time` BETWEEN "00:00:00" AND "12:00:00" THEN "Morning"
			WHEN `time` BETWEEN "12:01:00" AND "16:00:00" THEN "Afternoon"
			ELSE "Evening"
		END
    ) AS time_of_day
FROM sales;

-- create time of day column
ALTER TABLE sales ADD COLUMN time_of_day VARCHAR(20);

-- insert data into time of day column

UPDATE sales
	SET time_of_day = (
		CASE
			WHEN `time` BETWEEN "00:00:00" AND "12:00:00" THEN "Morning"
			WHEN `time` BETWEEN "12:01:00" AND "16:00:00" THEN "Afternoon"
			ELSE "Evening"
		END
    );
    
-- -------------------------------------------------------------------------------------------------------------

-- day name

-- test the column
SELECT date,DAYNAME(`date`) AS day_name
FROM sales;

-- create time of day column
ALTER TABLE sales ADD COLUMN day_name VARCHAR(10);

-- insert data into time of day column

UPDATE sales
	SET day_name = DAYNAME(`date`);

-- -------------------------------------------------------------------------------------------------------------

-- month name

-- test the column
SELECT date,MONTHNAME(`date`) AS day_name
FROM sales;

-- create time of day column
ALTER TABLE sales ADD COLUMN month_name VARCHAR(10);

-- insert data into time of day column

UPDATE sales
	SET month_name = MONTHNAME(`date`);
    
-- -------------------------------------------------------------------------------------------------------------	
-- EDA

-- How many unique cities does the data have?
SELECT COUNT(DISTINCT city) AS cities FROM sales;

-- In which city is each branch?
SELECT COUNT(DISTINCT branch) AS '#cities' FROM sales;

-- How many unique product lines does the data have?
SELECT COUNT(DISTINCT product_line) AS '#products' FROM sales;

-- What is the most common payment method?
select payment,count(payment) as count from sales group by payment order by count desc limit 1;

-- What is the most selling product line?
select product_line,count(product_line) as count from sales group by product_line order by count desc limit 1;

-- What is the total revenue by month?
select month_name, sum(total) as total_revenue from sales group by month_name order by total_revenue desc;

-- What month had the largest COGS?
select month_name, max(cogs) as largest_COGS from sales group by month_name order by largest_COGS desc limit 1;


-- What product line had the largest revenue?
select product_line, sum(total) as total_revenue from sales group by product_line order by total_revenue desc limit 1;

-- What is the city with the largest revenue?
select city, sum(total) as total_revenue from sales group by city order by total_revenue desc limit 1;

-- What product line had the largest VAT?
select product_line, sum(tax_pct) as total_vat from sales group by product_line order by total_vat desc limit 1;

-- Fetch each product line and add a column to those product line showing "Good", "Bad". Good if its greater than average sales

select product_line, sum(total) as avg_qnty, (
case
	when  sum(total)> (select avg(total) from sales) then "good"
    else "bad"
end
) as review
from sales group by product_line;

-- Which branch sold more products than average product sold?
select branch, sum(quantity) as total_sale_qnty from sales group by branch having total_sale_qnty>(select avg(quantity) from sales);

-- What is the most common product line by gender? 
(select gender,count(gender)as count,product_line  from sales group by gender,product_line having gender="male" order by count desc limit 1)
union 
(select gender,count(gender)as count,product_line  from sales group by gender,product_line having gender="female" order by count desc limit 1);

-- What is the average rating of each product line?
select product_line, round(avg(rating),2) as average_rating from sales group by product_line order by average_rating desc;

-- Number of sales made in each time of the day
select time_of_day, count(total) from sales group by time_of_day;

-- Which of the customer types brings the most revenue?
select customer_type, sum(total) as total_revenue from sales group by customer_type order by total_revenue desc limit 1;

-- Which city has the largest tax percent/ VAT (Value Added Tax)?
select city, max(tax_pct) largest_tax from sales group by city order by largest_tax desc limit 1;
 
-- Which customer type pays the most in VAT?
select customer_type, round(sum(tax_pct),2) as total_tax from sales group by customer_type order by total_tax desc limit 1;

-- How many unique customer types does the data have?
select count(distinct customer_type) from sales;

-- How many unique payment methods does the data have?
select count(distinct payment) from sales;

-- What is the most common customer type?
select customer_type, count(*) as  count from sales group by customer_type order by count desc limit 1;

-- Which customer type buys the most?
select customer_type, sum(quantity) as  count from sales group by customer_type order by count desc limit 1;

-- What is the gender of most of the customers?
select gender, count(*) as  count from sales group by gender order by count desc limit 1;

-- What is the gender distribution per branch?
select branch, gender, count(*) from sales group by gender, branch order by branch;

-- Which time of the day do customers give most ratings?
select time_of_day, round(sum(rating),2) as total_rating from sales group by time_of_day order by total_rating desc limit 1;

-- Which time of the day do customers give most ratings per branch?
select branch, time_of_day, round(sum(rating),2) as total_ratings from sales group by time_of_day, branch order by branch, total_ratings desc;

-- Which day fo the week has the best avg ratings?
select day_name, avg(rating) as avg_rating from sales group by day_name order by avg_rating desc limit 1;

-- Which day of the week has the best average ratings per branch?
(select branch,day_name,max(rating) as best_rating from sales group by day_name,branch having branch="A" order by branch,best_rating desc limit 1)
union
(select branch,day_name,max(rating) as best_rating from sales group by day_name,branch having branch="B" order by branch,best_rating desc limit 1)
union
(select branch,day_name,max(rating) as best_rating from sales group by day_name,branch having branch="C" order by branch,best_rating desc limit 1)