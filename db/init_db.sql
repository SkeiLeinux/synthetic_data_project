CREATE TABLE raw_data (
    id SERIAL PRIMARY KEY,
    column_1 VARCHAR,
    column_2 INTEGER,
    column_3 FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO raw_data (column_1, column_2, column_3) VALUES
('test1', 10, 0.5),
('test2', 20, 1.5),
('test3', 30, 2.5);
