import React, { Component, useState } from 'react';
import { Buffer } from "buffer";
import Accordion from 'react-bootstrap/Accordion';
import Collapse from 'react-bootstrap/Collapse';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import InputGroup from 'react-bootstrap/InputGroup';
import Pagination from 'react-bootstrap/Pagination';

import {ListObjectsV2Command, GetObjectCommand } from "@aws-sdk/client-s3"; 
import { InvokeModelWithResponseStreamCommand } from "@aws-sdk/client-bedrock-runtime";
import {getSignedUrl} from "@aws-sdk/s3-request-presigner"

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';

import './VideoTable.css';

export class VideoTable extends Component {
  constructor(props) {
    super(props);
    this.state = {
      videos: [],
      pages: {},
      firstFetch: false,
      searchText: "",
    };
    this.s3Client = props.s3Client
    this.bedrockClient = props.bedrockClient
    this.bucketName = props.bucketName
    this.modelId = props.modelId
    this.rawFolder = props.rawFolder
    this.summaryFolder = props.summaryFolder
    this.transcriptionFolder = props.transcriptionFolder
    this.videoScriptFolder = props.videoScriptFolder
    this.entitySentimentFolder = props.entitySentimentFolder

    this.renderVideoTableBody = this.renderVideoTableBody.bind(this)
    this.render = this.render.bind(this)
    this.renderChat = this.renderChat.bind(this)
    
  }

  async componentDidMount() {
    var [videos, pages] = await this.fetchVideos(0)
    this.setState({videos: videos, pages: pages, firstFetch: true})
  }
  
  async fetchVideos(page){
    var videos = []

    var token = null
    if(page > 0) token = this.state.pages[page].token;
    
    const listS3Input = {
      Bucket: this.bucketName,
      MaxKeys: Number("25"),
      Prefix: this.rawFolder + "/" + this.state.searchText,
      ContinuationToken: token
    };
    const listS3Command = new ListObjectsV2Command(listS3Input);
    const response = await this.s3Client.send(listS3Command);

    if(response.KeyCount == 0) return [videos, this.state.pages];

    for (const i in response.Contents){
      const vid = response.Contents[i]
      if (vid.Size <= 0) continue;

      // Get video name
      const name = vid.Key.replace(new RegExp("^"+ this.rawFolder +"\/", "g"),"")

      videos.push({
        index: videos.length,
        name: name,
        loaded: false,
        summary: undefined,
        videoScript: undefined,
        videoScriptShown: false,
        chats: [],
        url: undefined
      })
    }

    var pages = this.state.pages
    var pageFound = false
    for (const pi in pages){
      if(pages[pi].index == page){
        pageFound = true
        pages[pi] = {
          displayName: (page + 1).toString(),
          token: token,
          active: true,
          index: page
        }
      }else{
        pages[pi].active = false
      }
    }
    if(!pageFound){
      pages[page] = {
        displayName: (page + 1).toString(),
        token: token,
        active: true,
        index: page
      }
    }

    if (response.IsTruncated && 'NextContinuationToken' in response){
      pages[page+1] = {
        displayName: (page + 2).toString(),
        token: response.NextContinuationToken,
        active: false,
        index: page + 1
      } 
    }
    
    return [videos, pages]
  }

  async fetchVideoDetails(video){
    const name = video.name

    // Try to find the summary
    var command = new GetObjectCommand({
      Bucket: this.bucketName,
      Key: this.summaryFolder + "/" + name + ".txt",
    });
    try {
      const response = await this.s3Client.send(command);
      video.summary = await response.Body.transformToString();
    } catch (err) {}

    // Generate presigned URL for video
    const getObjectParams = {
      Bucket: this.bucketName,
      Key: this.rawFolder + "/" + name
    }
    command = new GetObjectCommand(getObjectParams);
    video.url = await getSignedUrl(this.s3Client, command, { expiresIn: 180 });

    // Try to get the video script
    var command = new GetObjectCommand({
      Bucket: this.bucketName,
      Key: this.videoScriptFolder + "/" + name + ".txt",
    });
    try {
      const response = await this.s3Client.send(command);
      video.videoScript = await response.Body.transformToString();
    } catch (err) {}

    video.loaded = true

    return video
  }

  async videoClicked(video) {
    if(!video.loaded) {
      video = await this.fetchVideoDetails(video)
      this.setState({videos: this.state.videos})
    }
  }

  handleChatInputChange(video, event) {
    const chatText = event.target.value
    if(event.keyCode == 13) { // When Enter is pressed
      video.chats.push({
        actor: "You",
        text: chatText
      })
      this.setState({videos: this.state.videos})
      event.target.value = "" // Clear input field
      this.retrieveChatResponse(video.index)
    }
  }

  renderChat(video) {
    return video.chats.map((chat, i) => { 
        return (
          <Row><Col xs={1}><p align="left">{(chat.actor + " : ")}</p></Col><Col><p align="left">{chat.text}</p></Col></Row>
        )
    })
  }

  async retrieveChatResponse(videoIndex, target){
    var video = this.state.videos[videoIndex]
    if(video.chats.length <= 0) return // Ignore if no chats are there
    if(video.chats[video.chats.length - 1].actor != "You") return // Ignore if last message was from bot

    var prompt = ""
    prompt += "\n\nHuman: You are an expert in analyzing video and you can answer questions about the video given the video script.\n\n"
    prompt += "Below is the video script with information about the voice heard in the video, the visual scenes seen, and the visual text visible."
    prompt += "The numbers on the left represents the seconds into the video where the information was extracted.\n"
    prompt += "<VideoScript>\n" + video.videoScript + "</VideoScript>\n\n"


    for (const i in video.chats){
      if (video.chats[i].actor == "You"){
        if(video.chats.length == 1) {
          prompt += video.chats[i].text
        }else{
          prompt += "\n\nHuman: " + video.chats[i].text
        }
      }else{
        prompt += "\n\nAssistant: " + video.chats[i].text
      }
    }
    prompt += "\n\nAssistant:"

    const payload = JSON.stringify({
      prompt: prompt,
      max_tokens_to_sample: 1000,
    })

    const input = { 
      body: payload, 
      modelId: this.modelId, 
      contentType: "application/json",
      accept: "application/json"
    };
    const command = new InvokeModelWithResponseStreamCommand(input);
    const response = await this.bedrockClient.send(command);

    video.chats.push({actor: "Bot", text: ""})

    for await (const event of response.body) {
        if (event.chunk && event.chunk.bytes) {
            const chunk = JSON.parse(Buffer.from(event.chunk.bytes).toString("utf-8"));
            video.chats[video.chats.length - 1].text += chunk.completion
            this.setState({videos: this.state.videos})
        }
    };
  }

  async handleSearchTextChange(event){
    this.setState({searchText: event.target.value })
  }

  async handleSearch(){
    await this.setState({pages: []})
    var [videos, pages] = await this.fetchVideos(0)
    this.setState({videos: videos, pages: pages})
  }
  
  renderVideoTableBody() {
    // Get which page is active to make sure Accordion keeps open for same page and will auto close for page switch.
    var activePage = 0
    Object.keys(this.state.pages).forEach((i) => {if(this.state.pages[i].active) activePage = i})

    // Render the video names as Accordion headers.
    return this.state.videos.map((video, i) => { 
        return (
          <Accordion.Item eventKey={video.index.toString() + activePage.toString()}>
            <Accordion.Header onClick={this.videoClicked.bind(this, video)}>{video.name}</Accordion.Header>
            <Accordion.Body>
              <Row>
                { video.loaded  ? <Col><video width="100%" controls><source src={video.url} type="video/mp4" /></video></Col> : "" }
              </Row>
              <Row>
                <Col className={video.loaded  ? "" : "d-none"}>
                  <Row><Col><h5 align="left">Summary:</h5></Col></Row>
                  <Row><Col><p align="left">{typeof video.summary === "undefined" ? "Summary is not ready yet" : video.summary }</p></Col></Row>
                </Col>
              </Row>
              <Row>
                <Col className={video.loaded  ? "" : "d-none"}>
                  <Row><Col><h5 align="left">Video script:</h5></Col></Row>
                  <Row><Col>
                    <p align="left">
                      <Button variant="info" size="sm"
                        onClick={() => { video.videoScriptShown = !video.videoScriptShown; this.setState({videos: this.state.videos });}}
                        aria-expanded={video.videoScriptShown}
                      >
                        Show/hide
                      </Button>
                    </p>
                    <Collapse in={video.videoScriptShown}>
                      <p id={("video-" + video.index + "-script")} className="preserve-line-breaks" align="left">{typeof video.videoScript === "undefined" ? "Video script is not ready yet" : video.videoScript }
                      </p></Collapse>
                  </Col></Row>
                </Col>
              </Row>
              <Row>
                <Col className={(video.loaded && typeof video.videoScript !== "undefined" ) ? "" : "d-none"}>
                  <Row><Col><h5 align="left">Ask about the video:</h5></Col></Row>
                    {this.renderChat(video)}
                  <Row><Col>
                    <Form.Control
                      type="text"
                      id={("video-" + video.index + "-chat-input")}
                      placeholder="e.g. How many people are featured in this video?"
                      onKeyDown={this.handleChatInputChange.bind(this, video)}
                    />
                    <Form.Text id={("video-" + video.index + "-chat-input")} ></Form.Text>
                  </Col></Row>
                </Col>
              </Row>
            </Accordion.Body>
          </Accordion.Item>
        ) 
      })
       
  }

  async handlePageClick(index){
    var [videos, pages] = await this.fetchVideos(parseInt(index))
    this.setState({videos: videos, pages: pages})
  }

  renderPagination(index){
    var page = this.state.pages[index]
    return (
      <Pagination.Item onClick={this.handlePageClick.bind(this, index)} key={index} active={page.active}>{page.displayName}</Pagination.Item>
    )
  }
  
  render() {
    return (
      <Row>
        <Col>
          <Row>
            <Col>
              <InputGroup className="mb-3">
                <Form.Control
                  type="text"
                  placeholder="Search video by name or name prefix"
                  aria-label="Search"
                  onChange={this.handleSearchTextChange.bind(this)}
                  value={this.state.searchText}
                />
                <Button onClick={this.handleSearch.bind(this)} variant="success" id="search-video">
                  Search
                </Button>
              </InputGroup>
            </Col>
            {/* <Col></Col> */}
          </Row>
          <Row><Col>         
            <Pagination>
              { Object.keys(this.state.pages).map((i) => { return this.renderPagination(i) })}
            </Pagination>
          </Col></Row>
          <Row><Col>
            <Accordion>
              { Object.keys(this.state.pages).length > 0 && this.state.videos.length > 0 ? this.renderVideoTableBody() : ""}
              { this.state.firstFetch && Object.keys(this.state.pages).length == 0 && this.state.videos.length == 0 ? <p>Found no relevant video in S3 "{this.rawFolder}" folder</p> : ""}
            </Accordion>
          </Col></Row>
        </Col>
      </Row>
    )
  }
}