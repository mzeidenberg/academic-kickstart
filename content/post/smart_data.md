---
title: Smart Data
date : 2019-12-13
draft: false

---

Most data tables that are commonly used by analysts are what I call "dumb data" because the data often has no explicit 
semantics attached to it.

If you attach semantics, you can enforce rules. For instance, if we have a table of people, where some of the people 
in the table are parents of some of the other people in the table, and the dates birth and of death (for those that
are not currently living) are marked in the table, then we can enforce, for instance, the following semantic rules:

1. Each person's date of death must be on or after their date of birth (if they are not still alive)
2. Each person's mother's date of death must be on or later than their date of birth, if the mother is not still alive
3. Each person's father's date of death must be on or later than about nine months prior than that person's date of birth
4. A person must have exactly one biological mother and one biological father.
5. If a person appears to have lived significantly more than 100 years, that raises questions about the integriy
6. Dates in the future are not possible for dates of birth or death.
7. Dates in the distant past may be errors.
8. If a mother is recorded as being over 50 or under 10 at a child's birth, that data is questionable.

Consider places. Suppose we have a list of places with their areas and populations, where some of the places are parts
of other places in the list. These are some of the semantic rules that apply at any fixed point of time:

1. The population and area of a constituent place has to be less than or equal to that of the place that it is a part of it
2. If a set of places together completely make up another place (for instance, in the way that the five boroughs of New York City 
completely constitute the city), then their populations and areas need to add up to that of the place they constitute.

Often data sets violate their semantics, typically due to data entry or other recording errors, this results in an exercise 
known as data cleaning. If semantics were imposed at the time of data collection, then the need for data cleaning will be 
greatly reduced. For instance, if the system knows that the population of New York City in 2018 was about 8.4 million people,
if you attempt to enter a population for Brooklyn (one of New York City's five borough) of 11.8 million, the system should reject
it and tell you that either the 8.4 or 11.8 figure is wrong.

Consider rates. For instance, consider the unemployment rate in each of the fifty states and the nation as a whole. The 
national unemployment can't be higher than that of the state with the highest unemployment, and can't be lower than that 
of the state with the lowest unemployment.

All of these rules can be implemented in code to make data tables smarter.

Typically, rows in a data table represent entities in the real world, and the columns represent properties of those entities.
This can be translated into a propositional logic. So we might have a table of U.S. presidents with their birth and death dates. This would lead to, for instance, a propositional form like the following:

birth-date("Abraham Lincoln")  == "February 12, 1809"  
death-date("Abraham Lincoln") == "April 15, 1865"  
mother("Abraham Lincoln") == "Nancy Hanks Lincoln"  
birth-date("Nancy Hanks Lincoln")  == "February 5, 1784"  
death-date("Nancy Hanks Lincoln") == "October 5, 1818"

We can then express semantic rules like those listed above as in the following example:

birth-date(X) <= death-date(X)  
birth-date(X) <= death-date(mother(X)) 

We can see that in this case of Lincoln these rules hold, but if there was an error in the data, for instance, if Lincoln's 
mother was recorded as dying in 1808 (before Lincoln's birth in 1809) instead of 1818, the rules would detect a problem in the 
data and we would be able to infer that either Lincoln's birth date or Lincoln's mother's death date were recorded incorrectly.

In order to make this work, each column in the table needs to be strongly typed with its semantics. The birth and death date 
columns need to be labeled as dates, and the name column has to have the semantics of a person.
