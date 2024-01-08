---
layout: post
title: Go
date: 2023-12-13
description: A cheat sheet for Golang
tags: swe 
categories: software-engineering
giscus_comments: false
featured: false
related_posts: false
toc:
  sidebar: left
---
Go, also known as Golang, is a programming language developed by Google with a focus on simplicity, efficiency, and concurrency. Its clean syntax and static typing contribute to code readability and early error detection. Go promotes concurrent programming through lightweight threads called goroutines and channels for communication between them. The language features automatic garbage collection, reducing the burden on developers for memory management. Go's standard library is robust, covering a wide range of tasks, and the language is cross-platform, allowing for seamless deployment across different operating systems. With a compiled nature, Go produces a single binary executable, simplifying distribution.

## Basics
create a simple program and run.
```go
package main

import "fmt"

func main() {
  message := greetMe("world")
  fmt.Println(message)
}

func greetMe(name string) string {
  return "Hello, " + name + "!"
}
```

```bash
go run hello.go
```

### Built-in data types
```markdown
| ------------ | --------------------------------------------- |
| String       | Numbers                                       | 
| ------------ | --------------------------------------------- |
| str = "hello"| byte, rune (char)                             |
| typname 'str'| int, int8, int16, int32, int64                |
|              | uint, uint8, uint16, uint32, uint64,  uintptr |
|              | float32, float64, complex64, complex128       |
| ------------ | --------------------------------------------- |
```

### Type conversion
```go
var i int = 42
var f float64 = float64(i)
var u uint = uint(f)
// alternative syntax
i := 42
f := float64(i)
u := uint(f)
```

## Control structure 
### if cond.
```go
func main() {
 // Basic one
 if x > 0 {
 return x
 } else {
 return -x
 }
 // You can put one statement before the condition
 if a := b + c; a < 42 {
 return a
 } else {
 return a - 42
 }
 // Type assertion inside if
 var val interface{}
 val = "foo"
 if str, ok := val.(string); ok {
 fmt.Println(str)
 }
}

```
### Loop
```go
// There's only `for`. No `while`, no `until`
 for i := 1; i < 10; i++ {
 }
 for ; i < 10; { // while loop
 }
 for i < 10 { // can omit semicolons if there's only a condition
 }
 for { // can omit the condition ~ while (true)
 }
```

### Switch
```go 
 // switch statement
 switch operatingSystem {
 case "darwin":
 fmt.Println("Mac OS Hipster")
 // cases break automatically, no fallthrough by default
 case "linux":
 fmt.Println("Linux Geek")
 default:
 // Windows, BSD, ...
 fmt.Println("Other")
 }
 // As with for and if, an assignment statement before the
 // switch value is allowed
 switch os := runtime.GOOS; os {
 case "darwin": ...
 }
```

## Array and slices 

```go
var a [10]int // int array with length 10. Length is part of type!
a[3] = 42 // set elements
i := a[3] // read elements
// declare and initialize
var a = [2]int{1, 2}
a := [2]int{1, 2} //shorthand
a := [...]int{1, 2} // elipsis -> Compiler figures out array length
```

```go
Slices
var a []int // a slice – like an array, but length is unspecified
var a = []int {1, 2, 3, 4} // declare and initialize a slice
 // (backed by given array implicitly)
a := []int{1, 2, 3, 4} // shorthand
chars := []string{0:"a", 2:"c", 1:"b"} // ["a", "b", "c"]
var b = a[lo:hi] // creates a slice (view of the array) from
 // index lo to hi-1
var b = a[1:4] // slice from index 1 to 3
var b = a[:3] // missing low index implies 0
var b = a[3:] // missing high index implies len(a)
```
```go
// create a slice with make
a = make([]byte, 5, 5) // first arg length, second capacity
a = make([]byte, 5) // capacity is optional

```
```go
// create a slice from an array
x := [3]string{"Rango", "Maven", "Java"}
s := x[:] // a slice referencing the storage of x

Operations on Arrays and Slices

len(a) //gives you the length of an array/a slice. It's a built-in function, not a attribute method
on the array.
// loop over an array/a slice
for i, e := range a {
 // i is the index, e the element
}
// if you only need e:
for _, e := range a {
 // e is the element
}
// ...and if you only need the index
for i := range a {
}
// In Go pre-1.4, it is a compiler error to not use i and e.
// Go 1.4 introduced a variable-free form:
for range time.Tick(time.Second) {
 // do it once a sec
}
```

## Maps
```go
var m map[string]int //{key(strint): value(int)}
m = make(map[string]int)
m["key"] = 42
fmt.Println(m["key"])
delete(m, "key")
elem, ok := m["key"] // test if key "key" is present, retrieve if so
// map literal
var m = map[string]Vertex{
 "Bell Labs": {40.68433, -74.39967},
 "Google": {37.42202, -122.08408},
}

```



## Structs
In go, the concept of class does not exist, but struct can have methods.
```go
// A struct is a type. It's also a collection of fields
// Declaration
type Vector struct {
 X, Y int
}
// Creating
var v = Vector{1, 2}
var v = Vector{X: 1, Y: 2} // Creates a struct by defining values
 // with keys
// Accessing members
v.X = 4
// You can declare methods on structs. The struct you want to declare
// the method on (the receiving type) comes between the func keyword
// and the method name. The struct is copied on each method call(!)
func (v Vector) Abs() float64 {
 return math.Sqrt(v.X*v.X + v.Y*v.Y)
}
// Call method
v.Abs()
// For mutating methods, you need to use a pointer (see below) to the
// Struct as the type. With this, the struct value is not copied for
// the method call.
func (v *Vector) add(n float64) {
 v.X += n
 v.Y += n
}

// Anonymous structs
Cheaper and safer than using map[string]interface{}.
point := struct {
 name, email string
}{"delta", "delta.d@mail.com"}
```

## Pointers
```go
p := Vector{0, 1} // p is a Vector
q := &p // q is a pointer to a Vector
r := &Vector{2, 3} // r is also a pointer to a Vector
// The type of a pointer to a Vector is *Vector
var s *Vector = new(Vector) // create ptr to a new struct instance
```

## Interface 
```go 
// interface declaration
type Delta interface {
 Unit() float64
 Division() string 
}
// types Operation *not* declare to implement interfaces

type Operation struct {
  commander string
  obj_one float64
  obj_two int
}
/* instead, types implicitly satisfy an interface if they implement all 
required methods*/
func (op Operation) Unit() float64 {
  return op.obj_one * obj_two
}

func (op Operation) Division() string {
  return "Air"
}

```

## Embedding 
There is no inheritance in Go. Instead, there is interface and struct embedding (composition).
```go

// ReadWriter implementations must satisfy both Reader and Writer
type ReadWriter interface {
 Reader
 Writer
}
// Server exposes all the methods that Logger has
type Server struct {
 Host string
 Port int
 *log.Logger
}
// initialize the embedded type the usual way
server := &Server{"localhost", 80, log.New(...)}
// methods implemented on the embedded struct are passed through
server.Log(...) // calls server.Logger.Log(...)
// Field name of an embedded type is its type name ('Logger' here)
var logger *log.Logger = server.Logger

```

## Package
 - declared at the top of every source file
 - executables are in package main
 - upper case identifier: exported (visible from other packages)
 - lower case identifier: private (not visible from other packages)







## Functions
```go
// a simple function
func Operation() {}
// function with parameters (again, types go after identifiers)
func Operation(param1 string, param2 int) {}
// multiple parameters of the same type
func Operation(param1, param2 int) {}

// return type declaration
func Operation() int {
 return 42
}
// Can return multiple values at once
func returnMulti() (int, string) {
 return 42, "foobar"
}
var x, str = returnMulti()
// Return multiple named results simply by return
func returnMulti2() (n int, s string) {
 n = 42
 s = "foobar"
 // n and s will be returned
 return
}
var x, str = returnMulti2()

// named return 
func named_return(para int) (x, y int){
  x = para -1
  y = praa* 2

  return
}
```

### Functions as values and closures
```go

func main() {
 // assign a function to a name Lambda
 add := func(a, b int) int {
 return a + b
 }
 // use the name to call the function
 fmt.Println(add(3, 4))
}
// Closures, lexically scoped: Functions can access values that were
// in scope when defining the function
func scope() func() int{
 outer_var := 2
 foo := func() int { return outer_var}
 return foo
}
func another_scope() func() int{
 // won't compile - outer_var and foo not defined in this scope
 outer_var = 444
 return foo
}
// Closures: don't mutate outer vars, instead redefine them!
func outer() (func() int, int) {
 outer_var := 2 // NOTE outer_var is outside inner's scope
 inner := func() int {
 outer_var += 99 // attempt to mutate outer_var
 return outer_var // => 101 (but outer_var is a newly redefined
 // variable visible only inside inner)
 }
 return inner, outer_var // => 101, 2 (still!)
}
```

### Variadic functios 
```go 
func main() {
 fmt.Println(adder(1, 2, 3)) // 6
 fmt.Println(adder(9, 9)) // 18
 nums := []int{10, 20, 30}
 fmt.Println(adder(nums...)) // 60
}
// Using ... before the type name of the last parameter indicates
// that it takes zero or more of those parameters.
// The function is invoked like any other function except we can
// pass as many arguments as we want.
func adder(args ...int) int {
 total := 0
 for _, v := range args { // Iterate over all args
 total += v
 }
 return total
}

```

## Errors

There is no exception handling. Functions that might produce an error just declare an
additional return value of type Error. This is the Error interface:
```go
type error interface {
 Error() string
}
//A function that potentially returns an error:
 
func doStuff() (int, error) {
}
func main() {
 result, error := doStuff()
 if (error != nil) {
 // handle error
 } else {
 // do your stuff
 }
}
```

### Defer 
Defers running a function until the surrounding function returns. The arguments are evaluated immediately, but the function call is not ran until later. <a href="https://go.dev/blog/defer-panic-and-recover">defer, panic, and recover</a> 

```go 
func main() {
  defer func() {
    fmt.Println("Done")
  }()
  fmt.Println("Working...")
}
```
The following defer func uses current value of d, unless we use a pointer to get final value at end of main.
```go 
func main() {
  var d = int64(0)
  defer func(d *int64) {
    fmt.Printf("& %v Unix Sec\n", *d)
  }(&d)
  fmt.Print("Done ")
  d = time.Now().Unix()
}
```


## Concurrency 
Goroutines are lightweight threads (managed by Go, not OS threads). 
```go 
go func(x,y) //starts a new go routine which runs given funciton i.e func
```
```go 
// a function (which can be later started as a goroutine)
func doStuff(s string) {
}
func main() {
 // using a named function in a goroutine
 go doStuff("foobar")
 // using an anonymous inner function in a goroutine
 go func (x int) {
 // function body goes here
 }(42)
}
```

### Channels 
Channels are concurrency-safe communication objects, used in goroutines.

```go 
ch := make(chan int) // create a channel of type int

ch <- 42 // Send a value to the channel ch.

v := <-ch // Receive a value from ch
/*Non-buffered channels block. Read blocks when no value is available.
write blocks if a value already has been written but not read. 
```
#### Buffered Channels
Buffered channels limit the amount of messages it can keep.
``` go
/*Create a buffered channel. Writing to a buffered channels does not block if less than 
<buffer size> unread values have been written. */
ch := make(chan int, 100)

close(c) /* closes the channel (only sender should close)
 Read from channel and test if it has been closed
 If ok is false, channel has been closed */
v, ok := <-ch
```

#### WaitGroup
A WaitGroup waits for a collection of goroutines to finish. The main goroutine calls Add to set the number of goroutines to wait for. The goroutine calls ```wg.Done()``` when it finishes.

```go 

import "sync"

func main() {
  var wg sync.WaitGroup
  
  for _, item := range itemList {
    // Increment WaitGroup Counter
    wg.Add(1)
    go doOperation(&wg, item)
  }
  // Wait for goroutines to finish
  wg.Wait()
  
}

func doOperation(wg *sync.WaitGroup, item string) {
  defer wg.Done()
  // do operation on item
  // ...
}
```

#### Iteration
```go
/*Read from channel until it is closed*/
for i := range ch {
 fmt.Println(i)
}
```
```go
/* select blocks on multiple channel operations. If one unblocks, the corresponding case is executed */
func doStuff(channelOut, channelIn chan int) {
 select {
 case channelOut <- 42:
 fmt.Println("We could write to channelOut!")
 case x := <- channelIn:
 fmt.Println("We could read from channelIn")
 case <-time.After(time.Second * 1):
 fmt.Println("timeout")
 }
}
```

#### Channels axioms

```go
// 1. A send to a nil channel blocks forever
var c chan string
c <- "Hello, World!"
// fatal error: all goroutines are asleep - deadlock!

 // 2. A receive from a nil channel blocks forever
var c chan string
fmt.Println(<-c)
// fatal error: all goroutines are asleep - deadlock!

 // 3. A send to a closed channel panics
var c = make(chan string, 1)
c <- "Hello world"
close(c)

c <- "Hello, Panic!"
// panic: send on closed channel

// 5. A receive from a close channel returns the zero value immediately
var c = make(chan int, 2)
c <- 1
c <- 2
close(c)
for i := 0; i < 3; i++ {
 fmt.Printf("%d ", <-c)
}
// 1 2 0

```
## Resources 
- <a href="https://awesome-go.com/">Awesome Go </a>
- <a href="https://go.dev/doc/effective_go">Effective in Go</a> 
- <a href="https://gobyexample.com/"> Go by Example </a>
- <a href="https://go.dev/wiki/"> Go WiKi </a>
