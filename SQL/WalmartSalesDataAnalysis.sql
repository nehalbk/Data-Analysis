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