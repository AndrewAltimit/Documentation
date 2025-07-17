---
layout: docs
title: Database Fundamentals
section: technology
---

# Database Fundamentals

## Overview

A database is an organized collection of structured data stored electronically in a computer system. Database management systems (DBMS) provide interfaces for storing, retrieving, and managing data efficiently while ensuring data integrity, security, and concurrent access.

## Database Types

### Relational Databases (RDBMS)
Relational databases organize data into tables with rows and columns, using structured query language (SQL) for data manipulation.

**Characteristics:**
- Fixed schema with predefined structure
- ACID compliance (Atomicity, Consistency, Isolation, Durability)
- Strong consistency guarantees
- Relationships between tables via foreign keys

**Popular Systems:**
- PostgreSQL
- MySQL
- Oracle Database
- Microsoft SQL Server
- SQLite

### NoSQL Databases
NoSQL databases provide flexible schemas and are optimized for specific data models and access patterns.

**Document Stores:**
- Store data as documents (usually JSON/BSON)
- Examples: MongoDB, CouchDB, Amazon DocumentDB

**Key-Value Stores:**
- Simple key-value pair storage
- Examples: Redis, Amazon DynamoDB, etcd

**Column-Family Stores:**
- Store data in column families
- Examples: Apache Cassandra, HBase, Amazon Keyspaces

**Graph Databases:**
- Optimize for relationships between data
- Examples: Neo4j, Amazon Neptune, ArangoDB

## Core Concepts

### Tables and Relations
In relational databases, data is organized into tables:
- **Table**: Collection of related data entries
- **Row (Record)**: Single data entry
- **Column (Attribute)**: Property of data
- **Primary Key**: Unique identifier for each row
- **Foreign Key**: Reference to primary key in another table

### ACID Properties
- **Atomicity**: Transactions complete fully or not at all
- **Consistency**: Data remains valid according to defined rules
- **Isolation**: Concurrent transactions don't interfere
- **Durability**: Committed data persists through system failures

### CAP Theorem
Distributed systems can guarantee only two of:
- **Consistency**: All nodes see the same data
- **Availability**: System remains operational
- **Partition Tolerance**: System continues during network failures

## SQL Fundamentals

### Data Definition Language (DDL)
```sql
-- Create table
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alter table
ALTER TABLE customers ADD COLUMN phone VARCHAR(20);

-- Drop table
DROP TABLE customers;
```

### Data Manipulation Language (DML)
```sql
-- Insert data
INSERT INTO customers (name, email) 
VALUES ('John Doe', 'john@example.com');

-- Select data
SELECT * FROM customers WHERE email LIKE '%@example.com';

-- Update data
UPDATE customers SET phone = '123-456-7890' WHERE id = 1;

-- Delete data
DELETE FROM customers WHERE created_at < '2023-01-01';
```

### Joins
```sql
-- Inner join
SELECT orders.id, customers.name, orders.total
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id;

-- Left join
SELECT customers.name, COUNT(orders.id) as order_count
FROM customers
LEFT JOIN orders ON customers.id = orders.customer_id
GROUP BY customers.id;
```

## Indexing

Indexes improve query performance by creating data structures for faster lookups.

### Index Types
- **B-Tree**: Default, good for range queries
- **Hash**: Fast equality comparisons
- **Bitmap**: Efficient for low-cardinality columns
- **Full-text**: Text search optimization

```sql
-- Create index
CREATE INDEX idx_email ON customers(email);

-- Composite index
CREATE INDEX idx_name_email ON customers(name, email);

-- Unique index
CREATE UNIQUE INDEX idx_unique_email ON customers(email);
```

## Transactions

Transactions group multiple operations into atomic units:

```sql
BEGIN TRANSACTION;

UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- Commit if successful
COMMIT;

-- Or rollback on error
ROLLBACK;
```

## Normalization

Database normalization reduces redundancy and improves data integrity:

### Normal Forms
- **1NF**: Atomic values, no repeating groups
- **2NF**: 1NF + no partial dependencies
- **3NF**: 2NF + no transitive dependencies
- **BCNF**: 3NF + every determinant is a candidate key

### Example Normalization
```sql
-- Denormalized
CREATE TABLE orders (
    id INTEGER,
    customer_name VARCHAR(100),
    customer_email VARCHAR(255),
    product_name VARCHAR(100),
    product_price DECIMAL(10,2)
);

-- Normalized
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255)
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10,2)
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER
);
```

## Query Optimization

### Explain Plans
```sql
EXPLAIN SELECT * FROM orders WHERE customer_id = 123;
```

### Optimization Techniques
- Use appropriate indexes
- Avoid SELECT *
- Limit result sets
- Use prepared statements
- Denormalize for read-heavy workloads
- Partition large tables

## NoSQL Data Modeling

### Document Model (MongoDB)
```javascript
{
    "_id": ObjectId("..."),
    "name": "John Doe",
    "email": "john@example.com",
    "orders": [
        {
            "id": 1,
            "items": ["product1", "product2"],
            "total": 99.99
        }
    ]
}
```

### Key-Value Model (Redis)
```
SET user:1000:name "John Doe"
SET user:1000:email "john@example.com"
HSET user:1000:settings theme "dark" language "en"
```

## Best Practices

### Schema Design
- Define clear relationships
- Use appropriate data types
- Implement constraints
- Plan for growth

### Performance
- Index frequently queried columns
- Monitor query performance
- Use connection pooling
- Implement caching strategies

### Security
- Use parameterized queries
- Implement access controls
- Encrypt sensitive data
- Regular backups

### Maintenance
- Regular vacuuming/optimization
- Monitor disk usage
- Update statistics
- Plan for scaling

## References

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [Database Design Patterns](https://www.databasepatterns.com/)