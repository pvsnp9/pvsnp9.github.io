---
layout: post
title: Protobuf
date: 2023-12-31
description: A guide on Protobuf
tags: swe 
categories: software-engineering
giscus_comments: false
featured: true
related_posts: false
toc:
  sidebar: left
---
## Introduction 


Protocol Buffers (protobuf) is a language-agnostic data `byte` serialization format developed by Google. It uses a simple language to define data structures in `.proto` files. Protobuf supports various data types, including integers, floating-point numbers, booleans, strings, bytes, and custom message types. Messages, the fundamental building blocks, are collections of key-value pairs, akin to objects or structs in programming languages. Additionally, protobuf introduces advanced features such as enumerations, repeated fields, and maps. The format is designed for efficiency, producing compact binary representations of data. Code generators then transform these definitions into source code in different programming languages, allowing seamless data exchange between applications written in diverse languages while providing support for backward and forward compatibility. In addition , gRPC is an open-source RPC framework developed by Google that facilitates efficient, language-agnostic communication between distributed systems using Protocol Buffers

In this blog, we'll delve into the fundamentals of Protocol Buffers and explore the seamless integration of gRPC in the Go programming language, uncovering the power of these tools in enhancing communication and interoperability in modern software architectures.

JSON
```json 
{
    "id": 1,
    "name": "Delta",
    "unit": "Special",
    "rank": 1,
    "division": "Air"
}
```
Protobuf
```protobuf
syntax: "proto3";
option go_package ="{package_name}/pkg/pb/protogen";

package basic;

message Officer {
    uint32 id = 1;
    string name = 2;
    int rank = 3;
    string division = 4;
}

```
The `option go_package` tells the go package name and detination for proto generated source code. 


Sidenote: 
`Marshal` refers to the process of converting a data structure into its serialized representation, often in the form of a byte slice or a string. This operation is commonly used for data interchange or storage. Conversely, `Unmarshal` involves taking the serialized data and reconstructing the original data structure. These operations are frequently used in encoding and decoding data, such as converting between Go data structures and formats like JSON, XML, or Protocol Buffers. The encoding/json and other related packages in Go provide functions like json.Marshal and json.Unmarshal to perform these operations for JSON encoding and decoding, while similar functions exist for other formats and libraries.
```
Note: Marshall is serialization and Unmarshall is deserilization. 
```




Protobuf is designed with schema evolution in mind to avoid braeaking changes. It needs to follow a certain rules. 

Protobuf styling Guide<a href="https://github.com/uber/prototool/blob/dev/style/README.md#spacing"> (Uber) </a>



## Makefile 
```makefile
GO_MODULE := github.com/pvsnp9/example


.PHONY: tidy
tidy:
	go mod tidy


.PHONY: clean
clean:
	@if [ -d "protogen" ]; then rm -rf protogen; fi


.PHONY: proto
proto:
	protoc --go_opt=module=${GO_MODULE} --go_out=. ./pkg/pb/**/*.proto


.PHONY: build
build: clean protoc tidy


.PHONY: run
run:
	go run cmd/main.go


```




When you use the Protocol Buffers compiler (`protoc`) to generate code for a proto file, it typically produces files with certain naming conventions based on the target language and the type of service definition. Here are the meanings of the files you mentioned:

1. **`*.pb.go` Files:**
   - These are Go source files generated by the `protoc` compiler when targeting Go (Golang).
   - The `*.pb.go` files contain the generated code for your protocol buffer messages, allowing you to easily serialize and deserialize data structures defined in your proto file.
   - The file name is typically derived from your original proto file. For example, if your proto file is named `example.proto`, the generated Go file might be named `example.pb.go`.

2. **`*_grpc.pb.go` Files:**
   - When you define a gRPC service in your proto file, the `protoc` compiler generates additional code for the service implementation and client in addition to the message types.
   - The `*_grpc.pb.go` file contains the gRPC service interface and the server-side implementation code.
   - The naming convention usually involves adding `_grpc.pb.go` to the base name of your proto file. For example, if your proto file is named `example.proto`, the generated gRPC file might be named `example_grpc.pb.go`.

Here's a quick example to illustrate the naming conventions:

- Original Proto File: `example.proto`
- Generated Go File for Messages: `example.pb.go`
- Generated Go File for gRPC Service: `example_grpc.pb.go`

These files, once generated, allow you to work with your protocol buffer messages and gRPC service in your Go application. Remember that the actual names may depend on your specific proto file and project structure.


## Repeated fields, enumeration, Comments 

Repeated : A filed with one or zero values like a array/list. The default value is empty.

```protobuf
message Person {
    repeated int itemId
}
```


```protobuf
enum Gender {
    GENDER_UNSPECIFIED = 0;
    GENDER_MALE = 1;
    GENDER_FEMALE = 2;
}
```

How to use them:
```protobuf 
message User {
  uint32 id = 1;
  string username = 2;
  bool is_active = 3;
  repeated string emails = 4;
  Gender gender = 5;
}
```



## Nested message type

```protobuf
// Define another message called "Address"
message Address {
  // Define fields for the Address message
  string street = 1;
  string city = 2;
  string country = 3;
  string postal_code = 4;
  Coordinate coordinate = 15; //field to access the cordindate 

  // nested
  message Coordinate {
      double lattitude = 1;
      double longitude = 2;
  }
}
```

```go 
//Using it 
address := basic.Address{
    Street:     "Street",
    City:       "City",
    PostalCode: "PSTCD3",
    Coordinate: &basic.Address_Coordinate{
        Lattitude: 40.705152254,
        Longitude: -74.52515425,
    },
    }

```





##ProtoJSON

a tool to convert JSON to protobuf and vice-versa.

```go

import (
	"log"
	"github.com/pvsnp9/example/pkg/protogen/basic"
	"google.golang.org/protobuf/encoding/protojson"
)

func ProtoToJsonUser() {
	u := basic.User{
		Id:       2,
		Username: "Bravo",
		IsActive: true,
		Password: []byte("adfasfas"),
		Emails:   []string{"test@mai.com", "test2@mail.com"},
		Gender:   basic.Gender_GENDER_MALE,
	}
	jsonBytes, err := protojson.Marshal(&u)
	if err != nil {
		log.Println("error", err)
	}
	log.Println(string(jsonBytes))
}

func JsonToProto() {
	json := `{
		"id": 5,
		"username":"delta",
		"is_active":true,
		"password":"YWRmYXNmYXM=",
		"emails":["test@mai.com", "test2@mail.com"],
		"gender":"GENDER_MALE"
	}`

	var user basic.User
	err := protojson.Unmarshal([]byte(json), &user)
	if err != nil {
		log.Println("Err:", err)
	}
	log.Println(&user)
}


```



## Importing Protos 

```protobuf
syntax = "proto3";

import "pkg/pb/basic/user.proto";
// package_name/yoo=ur_directory

package basic;

option go_package = "github.com/pvsnp9/example/pkg/protogen/basic";
// package_name/generated file destiantion

message UserGroup {
    int32 group_id = 1 [json_name="group_id"];
    string group_name = 2 [json_name="group_name"];
    repeated string roles = 3;
    repeated User user = 4;
    string description = 5;
}

```
Importing non-local protos.

```protobuf 
syntax = "proto3";

import "pkg/pb/basic/application.proto";
import "pkg/pb/dummy/application.proto";

package jobsearch;

option go_package = "github.com/pvsnp9/example/pkg/protogen/jobsearch";

message JobCandidate {
    uint32 job_candidate_id = 1 [json_name = "job_candidate_id"];
    dummy.Application application = 2;
  }
  
  message JobSoftware {
    uint32 job_software_id = 1 [json_name = "job_software_id"];
    basic.Application application = 2;
  }

```

## Any 

```protobuf
  message Papermail{
    string paper_mail_address = 1 [json_name = "paper_mail_address"];
  }

  message SocialMedia{
    string social_media_platform = 1 [json_name = "social_media_platform"];
    string social_media_username = 2 [json_name = "social_media_username"];
  }

  message InstantMessaging {
    string instant_messaging_product = 1 [json_name = "instant_messaging_product"];
    string instant_messaging_username = 2 [json_name = "instant_messaging_username"];
  }
```
```protobuf
//usage 
message User {
     google.protobuf.Any communication_channel = 19 [json_name = "communication_channel"];
}
```
```go
//examples 
func randomCommunicationChanel() anypb.Any {
	paper_mail := basic.Papermail{
		PaperMailAddress: "Mailing address !!",
	}
	social_media := basic.SocialMedia{
		SocialMediaPlatform: "insta",
		SocialMediaUsername: "rango",
	}

	instant_messaging := basic.InstantMessaging{
		InstantMessagingProduct:  "whatsapp",
		InstantMessagingUsername: "Rango",
	}

	var any anypb.Any

	switch r := rand.Intn(20) % 3; r {
	case 0:
		anypb.MarshalFrom(&any, &paper_mail, proto.MarshalOptions{})
	case 1:
		anypb.MarshalFrom(&any, &social_media, proto.MarshalOptions{})
	default:
		anypb.MarshalFrom(&any, &instant_messaging, proto.MarshalOptions{})
	}
	return any
}

// if we know which message type it is.
func BasicUnMarshallAnyToKnown() {
	sm := basic.SocialMedia{
		SocialMediaPlatform: "Fb",
		SocialMediaUsername: "tt",
	}
	var a anypb.Any
	anypb.MarshalFrom(&a, &sm, proto.MarshalOptions{})
	smedia := basic.SocialMedia{}

	if err := a.UnmarshalTo(&smedia); err != nil {
		return
	}
	json, _ := protojson.Marshal(&sm)
	log.Print(string(json))
}

//if we do not know message type
func BasicUnMarshallAnyToUnknown() {
	anon := randomCommunicationChanel()

	var anonUnmarshalled protoreflect.ProtoMessage

	anonUnmarshalled, err := anon.UnmarshalNew()
	if err != nil {
		return
	}
	log.Print("Unmarshall as a ", anonUnmarshalled.ProtoReflect().Descriptor().Name())
	json, _ := protojson.Marshal(anonUnmarshalled)
	log.Print(string(json))
}


```
## Oneof 
Allow only X, Y, or custom_filed. It uses $oneof$ keyword to define.

```protobuf 
message User {
    oneof electronic_comm_channel {
        SocialMedia social_media = 20 [json_name = "social_media"];
        InstantMessaging instant_messaging = 21 [json_name = "instant_messaging"];
    }
}

```

```go 
// oneof example
func BasicOneof() {
	sm := basic.SocialMedia{
		SocialMediaPlatform: "X",
		SocialMediaUsername: "Charlie",
	}

	ecom_chan := basic.User_SocialMedia{
		SocialMedia: &sm,
	}

	user := basic.User{
		Id:                    11,
		Username:              "LaLa",
		IsActive:              true,
		Password:              []byte("Rango"),
		Gender:                basic.Gender_GENDER_MALE,
		Emails:                []string{"test@mai.com", "test2@mail.com"},
		ElectronicCommChannel: &ecom_chan,
	}
	json, _ := protojson.Marshal(&user)
	log.Println(string(json))
}
```

## Map 
Its a key-value data structure. Protobuf also supports map data structure.
```protobuf
message User {
    map <string, uint32> skill_rating = 22 [json_name="skill_rating"];
}
```
```go 
skill_rating := map[string]uint32{"swim": 8, "fly": 9, "drive": 10}
user := User{
    SkillRating: skill_rating,
}
```
## Read/Write to Disk
```go 


// write to file
func WriteUserToFile(user proto.Message, filename string) {
	bytes, _ := proto.Marshal(user)
	if err := ioutil.WriteFile(filename, bytes, 0644); err != nil {
		log.Fatalln("Error writing to file", err)
	}
	log.Println("writing to file completed")
}

//Reading from file
func ReadUserFromDisk(dest proto.Message, filename string) {

	log.Println("Reading file ", filename)
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatalln("Errot reading file", err)
		return
	}
	// var user basic.User
	if err := proto.Unmarshal(bytes, dest); err != nil {
		log.Fatalln("Error on unmarshalling user", err)
	}

	json, _ := protojson.Marshal(dest)
	log.Print(string(json))
}
```

## Read/Write as JSON
```go

//write as a json
func WriteAsJson(msg proto.Message, filename string) {
	jsonBytes, _ := protojson.Marshal(msg)
	if err := ioutil.WriteFile(filename, jsonBytes, 0644); err != nil {
		log.Fatalln("could not write to file ", err)
		return
	}
	log.Print("Writing file is completed.")
}

//read as jsom and print
func ReadAsJson(dest proto.Message, filename string) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatalln("Could not read file ", err)
		return
	}
	if err := protojson.Unmarshal(bytes, dest); err != nil {
		log.Fatalln("Unmarshalling failed ", err)
		return
	}
	json, _ := protojson.Marshal(dest)
	log.Println(string(json))
}
```

## Schema Evolution 
it is  to gracefully adapt to changes in the structure of serialized data over time, enabling compatibility and interoperability between different versions of the schema.
- Compatibility: sender and receiver might have different protobuf message definition
- Forward & Backward compatibility

`Rules`
- Do not change field number `message A { string email = 1;}` but, renaming is allowed because the serialization and deserialization depends on number.
- Removing fields is okay but must not be used in future. The field number and field names are better reserved. 
`message A { reserved 3; reserved "phone_number"; reserved 3 to 5; reserved 'x', 'y';}`
- During deletion, be aware of the fact old and new binary will have differences. 

```protobuf

message UserContent {
  uint32 user_content_id = 1 [json_name = "user_content_id"];
  string slug = 2;
  string title = 3;
  string html_content = 4 [json_name = "html_content"];
  uint32 author_id = 5 [json_name = "author_id"];
}
```

```go 
//read and write content in v1 
user_content := basic.UserContent{
UserContentId: 12,
    Slug:          "slug12",
    Title:         "Test content",
    HtmlContent:   "<h1>Heading 1</h1>",
    AuthorId:      32,
}

var dest basic.UserContent

services.WriteUserToFile(&user_content, "user_content_v1.bin")
services.ReadUserFromDisk(&dest, "user_content_v1.bin")
```

`update the message UserContent add new field category`
```protobuf
  message UserContent {
  uint32 user_content_id = 1 [json_name = "user_content_id"];
  string slug = 2;
  string title = 3;
  string html_content = 4 [json_name = "html_content"];
  uint32 author_id = 5 [json_name = "author_id"];
  string category = 6 ;
}
```
Forward compatibility 
- Read `user_content_v1.bin` using UserContent version 2  (writer is older / forward compatibility)
- Write user_content_v2.bin using UserContent version 2
- Read user_content_v2.bin using UserContent version 2


`update the message UserContent to following`
```protobuf
message UserContent {
  reserved 3, 5;
  uint32 user_content_id = 1 [json_name = "user_content_id"];
  string slug = 2;
  // string title = 3;
  string html_content = 4 [json_name = "html_content"];
  // uint32 author_id = 5 [json_name = "author_id"];
  string category = 6 ;
  string sub_category = 7 [json_name="sub_category"];
}
```

```go 
//readung and writing in V3
user_content := basic.UserContent{
    UserContentId: 12,
    Slug:          "slug12",
    HtmlContent:   "<h1>Heading 1</h1>",
    Category:      "S",
    SubCategory:   "AA",
}

var dest basic.UserContent

services.WriteUserToFile(&user_content, "user_content_v3.bin")
services.ReadUserFromDisk(&dest, "user_content_v3.bin")
```

## Additional types 
Download types from following links, and place them into your directory.
<a href="https://protobuf.dev/reference/protobuf/google.protobuf/"> Documentation</a>
<a href="https://github.com/googleapis/googleapis/tree/master/google/type">Source files</a>

Example:
`pkg/
    pb/
    google/
    types/ g.proto
`

import
```go
import "pkg/pb/google/type/date.proto";
import "pkg/pb/google/type/latlng.proto";

message User {
    google.type.Date birth_date  = 24 [json_name = "birth_date"];
    google.type.LatLng last_known_location = 25 [json_name = "last_known_location"];
}

//usage
user := basic.User{
    Id:                    11,
    Username:              "LaLa",
    SkillRating:           skill_rating,
    LastLogin:             timestamppb.Now(),
    BirthDate:             &date.Date{Year: 2000, Month: 5, Day: 27},
    LastKnownLocation: &latlng.LatLng{
        Latitude:  -6.29847717,
        Longitude: 106.8290577,
    },
}
```
## <a href="https://github.com/bufbuild/protoc-gen-validate">Validate</a> 
Comming soon !!