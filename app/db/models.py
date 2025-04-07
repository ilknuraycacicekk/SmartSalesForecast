from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, LargeBinary, SmallInteger, Boolean
from sqlalchemy.orm import relationship
from .database import Base


class Product(Base):
    __tablename__ = "products"

    product_id = Column(SmallInteger, primary_key=True, index=True)
    product_name = Column(String(40), nullable=False, index=True)
    supplier_id = Column(SmallInteger, ForeignKey("suppliers.supplier_id"))
    category_id = Column(SmallInteger, ForeignKey("categories.category_id"))
    quantity_per_unit = Column(String(20))
    unit_price = Column(Float)
    units_in_stock = Column(SmallInteger)
    units_on_order = Column(SmallInteger)
    reorder_level = Column(SmallInteger)
    discontinued = Column(Integer, nullable=False)

    orders = relationship("OrderDetail", back_populates="product")
    category = relationship("Category", back_populates="products")
    supplier = relationship("Supplier", back_populates="products")


class Category(Base):
    __tablename__ = "categories"

    category_id = Column(SmallInteger, primary_key=True, index=True)
    category_name = Column(String(15), nullable=False, index=True)
    description = Column(Text)
    picture = Column(LargeBinary)

    products = relationship("Product", back_populates="category")


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String(5), primary_key=True, index=True)
    company_name = Column(String(40), nullable=False, index=True)
    contact_name = Column(String(30))
    contact_title = Column(String(30))
    address = Column(String(60))
    city = Column(String(15))
    region = Column(String(15))
    postal_code = Column(String(10))
    country = Column(String(15))
    phone = Column(String(24))
    fax = Column(String(24))

    orders = relationship("Order", back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(SmallInteger, primary_key=True, index=True)
    customer_id = Column(String(5), ForeignKey("customers.customer_id"))
    employee_id = Column(SmallInteger, ForeignKey("employees.employee_id"))
    order_date = Column(DateTime)
    required_date = Column(DateTime)
    shipped_date = Column(DateTime)
    ship_via = Column(SmallInteger, ForeignKey("shippers.shipper_id"))
    freight = Column(Float)
    ship_name = Column(String(40))
    ship_address = Column(String(60))
    ship_city = Column(String(15))
    ship_region = Column(String(15))
    ship_postal_code = Column(String(10))
    ship_country = Column(String(15))

    customer = relationship("Customer", back_populates="orders")
    employee = relationship("Employee", back_populates="orders")
    shipper = relationship("Shipper", back_populates="orders")
    details = relationship("OrderDetail", back_populates="order")


class OrderDetail(Base):
    __tablename__ = "order_details"

    order_id = Column(SmallInteger, ForeignKey("orders.order_id"), primary_key=True)
    product_id = Column(SmallInteger, ForeignKey("products.product_id"), primary_key=True)
    unit_price = Column(Float, nullable=False)
    quantity = Column(SmallInteger, nullable=False)
    discount = Column(Float, nullable=False)

    order = relationship("Order", back_populates="details")
    product = relationship("Product", back_populates="orders")


class Employee(Base):
    __tablename__ = "employees"

    employee_id = Column(SmallInteger, primary_key=True, index=True)
    last_name = Column(String(20), nullable=False, index=True)
    first_name = Column(String(10), nullable=False, index=True)
    title = Column(String(30))
    title_of_courtesy = Column(String(25))
    birth_date = Column(DateTime)
    hire_date = Column(DateTime)
    address = Column(String(60))
    city = Column(String(15))
    region = Column(String(15))
    postal_code = Column(String(10))
    country = Column(String(15))
    home_phone = Column(String(24))
    extension = Column(String(4))
    photo = Column(LargeBinary)
    notes = Column(Text)
    reports_to = Column(SmallInteger, ForeignKey("employees.employee_id"))
    photo_path = Column(String(255))

    manager = relationship("Employee", remote_side=[employee_id], backref="subordinates")
    orders = relationship("Order", back_populates="employee")
    territories = relationship("EmployeeTerritory", back_populates="employee")


class Supplier(Base):
    __tablename__ = "suppliers"

    supplier_id = Column(SmallInteger, primary_key=True, index=True)
    company_name = Column(String(40), nullable=False, index=True)
    contact_name = Column(String(30))
    contact_title = Column(String(30))
    address = Column(String(60))
    city = Column(String(15))
    region = Column(String(15))
    postal_code = Column(String(10))
    country = Column(String(15))
    phone = Column(String(24))
    fax = Column(String(24))
    homepage = Column(Text)

    products = relationship("Product", back_populates="supplier")


class Shipper(Base):
    __tablename__ = "shippers"

    shipper_id = Column(SmallInteger, primary_key=True, index=True)
    company_name = Column(String(40), nullable=False)
    phone = Column(String(24))

    orders = relationship("Order", back_populates="shipper")


class Region(Base):
    __tablename__ = "region"

    region_id = Column(SmallInteger, primary_key=True)
    region_description = Column(String)

    territories = relationship("Territory", back_populates="region")


class Territory(Base):
    __tablename__ = "territories"

    territory_id = Column(String, primary_key=True)
    territory_description = Column(String)
    region_id = Column(SmallInteger, ForeignKey("region.region_id"))

    region = relationship("Region", back_populates="territories")
    employees = relationship("EmployeeTerritory", back_populates="territory")


class EmployeeTerritory(Base):
    __tablename__ = "employee_territories"

    employee_id = Column(SmallInteger, ForeignKey("employees.employee_id"), primary_key=True)
    territory_id = Column(String, ForeignKey("territories.territory_id"), primary_key=True)

    employee = relationship("Employee", back_populates="territories")
    territory = relationship("Territory", back_populates="employees")


class UsStates(Base):
    __tablename__ = "us_states"

    state_id = Column(SmallInteger, primary_key=True)
    state_name = Column(String(100))
    state_abbr = Column(String(2))
    state_region = Column(String(50)) 