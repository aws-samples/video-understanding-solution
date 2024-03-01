import React, { Component, useState } from 'react';
import { Buffer } from "buffer";
import Accordion from 'react-bootstrap/Accordion';
import Collapse from 'react-bootstrap/Collapse';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import InputGroup from 'react-bootstrap/InputGroup';
import Pagination from 'react-bootstrap/Pagination';
import Spinner from 'react-bootstrap/Spinner';
import DatePicker from "react-datepicker";

import {ListObjectsV2Command, GetObjectCommand } from "@aws-sdk/client-s3"; 
import { InvokeModelWithResponseStreamCommand, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import {getSignedUrl} from "@aws-sdk/s3-request-presigner"

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';

import './VideoTable.css';
import "react-datepicker/dist/react-datepicker.css";

export class VideoTable extends Component {
  constructor(props) {
    super(props);
    this.state = {
      videos: [],
      pages: {},
      firstFetch: false,
      pageLoading: false,
      searchByNameText: "",
      searchByStartDate: null,
      searchByEndDate: null,
      searchByAboutText: "",
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
    this.restApiUrl = props.restApiUrl
    this.videosApiResource = props.videosApiResource
    this.cognitoTs = props.cognitoTs
    this.maxCharactersForChat = 5000
    this.maxCharactersForVideoScript = 300000
    this.maxCharactersForSingleLLMCall = 100000
    this.estimatedCharactersPerToken = 4

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

    this.setState({
      pageLoading: true
    })

    // Construct filter
    var params = `?page=${page.toString()}&`
    if(this.state.searchByNameText != ""){
      params += `videoNameContains=${encodeURI(this.state.searchByNameText)}&`
    }
    if(this.state.searchByStartDate != null && this.state.searchByStartDate != null ){
      const startDateString = this.state.searchByStartDate.toISOString()
      const endDateString = this.state.searchByEndDate.toISOString()
      params += `uploadedBetween=${encodeURI(startDateString + "|" + endDateString)}&`
    }
    if(this.state.searchByAboutText != ""){
      params += `about=${encodeURI(this.state.searchByAboutText)}`
    }

    // Call API Gateway to fetch the video names
    var ts = await this.cognitoTs;

    const response = await fetch(`${this.restApiUrl}/${this.videosApiResource}${params}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': ts.idToken.toString(),
      }
    })

    const responseBody = await(response.json())
    
    if(responseBody.videos.length == 0) return [videos, this.state.pages];

    // Add the video representation in the UI
    for (const i in responseBody.videos){
      const videoName = responseBody.videos[i]

      videos.push({
        index: videos.length,
        name: videoName,
        loaded: false,
        summary: undefined,
        rawEntities: undefined,
        entities: undefined,
        videoScript: undefined,
        videoShown: false,
        chats: [],
        chatSummary: "",
        chatLength: 0,
        chatWaiting: false,
        url: undefined
      })
    }

    // Activate the right page on the pagination bar
    var pages = this.state.pages
    var pageFound = false
    for (const pi in pages){
      if(pages[pi].index == page){
        pageFound = true
        pages[pi] = {
          displayName: (page + 1).toString(),
          active: true,
          index: page
        }
      }else{
        pages[pi].active = false
      }
    }
    // If the page is not found, add it to the pagination bar
    if(!pageFound){
      pages[page] = {
        displayName: (page + 1).toString(),
        active: true,
        index: page
      }
    }

    // If there is a next page indicated in the API call return, then either add a new page in pagination bar or just update existing page metadata
    if ("nextPage" in responseBody){
      pages[page+1] = {
        displayName: (page + 2).toString(),
        active: false,
        index: page + 1
      } 
    }
    
    this.setState({
      pageLoading: false
    })

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
    } catch (err) {
      console.log(err)
    }

    // Try to find the entities and sentiment
    var command = new GetObjectCommand({
      Bucket: this.bucketName,
      Key: this.entitySentimentFolder + "/" + name + ".txt",
    });
    try {
      const response = await this.s3Client.send(command);
      const rawEntities = await response.Body.transformToString();
      video.rawEntities = rawEntities
      video.entities = []
      rawEntities.split("\n").forEach(entity => {
        const data = entity.split("|")
        video.entities.push(data)
      })
    } catch (err) {
      console.log(err)
    }


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

  async showVideo(video){
    // Generate presigned URL for video
    const getObjectParams = {
      Bucket: this.bucketName,
      Key: this.rawFolder + "/" + video.name
    }
    const command = new GetObjectCommand(getObjectParams);
    video.url = await getSignedUrl(this.s3Client, command, { expiresIn: 180 });
    video.videoShown = true
    this.setState({videos: this.state.videos})
  }

  async videoClicked(video) {
    if(!video.loaded) {
      video = await this.fetchVideoDetails(video)
      this.setState({videos: this.state.videos})
    }
  }

  handleChatInputChange(video, event) {
    const chatText = event.target.value
    if(event.keyCode == 13 && !event.shiftKey) { // When Enter is pressed without Shift
      video.chats.push({
        actor: "You",
        text: chatText
      })
      video.chatLength += chatText.length // Increment the length of the chat
      this.setState({videos: this.state.videos})
      event.target.value = "" // Clear input field
      this.handleChat(video.index)
    }
  }

  renderChat(video) {
    return video.chats.map((chat, i) => { 
        return (
          <Row key={`video-chat-${video.index}-${i}`}><Col xs={1}><p align="left">{(chat.actor + " : ")}</p></Col><Col><p align="left">{chat.text}</p></Col></Row>
        )
    })
  }

  async retrieveAnswerForShortVideo(video, systemPrompt){
    var [videoScriptPrompt, chatPrompt, closingPrompt] = ["","",""]

    videoScriptPrompt += "Below is the video timeline with information about the voice heard in the video, visual scenes seen, visual text visible, and any face or celebrity detected. "
    videoScriptPrompt += "The numbers on the left represents the seconds into the video where the information was extracted.\n"
    videoScriptPrompt += `<VideoTimeline>\n${video.videoScript}</VideoTimeline>\n\n`

    // If the chat history is not too long, then include the whole history in the prompt
    // Otherwise, use the summary + last chat instead.
    // However, when the summary is not yet generated (First generation is when characters exceed 5000), then use the whole history first
    // The last scenario mentioned above may happen during the transition of <5000 characters to >5000 characters for the history
    if (video.chatLength <= this.maxCharactersForChat || video.chatSummary == ""){ 
      for (const i in video.chats){
        if (video.chats[i].actor == "You"){
          if(video.chats.length == 1) {
            chatPrompt += video.chats[i].text
          }else{
            chatPrompt += "\n\nHuman: " + video.chats[i].text
          }
        }else{
          chatPrompt += "\n\nAssistant: " + video.chats[i].text
        }
      }
    }else{ // If the chat history is too long, so that we include the summary of the chats + last chat only.
      chatPrompt += "You had a conversation with a user about that video as summarized below.\n"
      chatPrompt += `<ChatSummary>\n${video.chatSummary}\n</ChatSummary>\n`
      chatPrompt += "Now the user comes back with a reply below.\n"
      chatPrompt += `<Question>\n${video.chats[video.chats.length - 1].text}\n</Question>\n`
      chatPrompt += "Answer the user's reply above directly, without any XML tag.\n"
    }

    closingPrompt += "\n\nAssistant:"

    const prompt = systemPrompt + videoScriptPrompt + chatPrompt + closingPrompt
    const input = { 
      body: JSON.stringify({
        prompt: prompt,
        max_tokens_to_sample: 1000,
        stop_sequences: ["\nUser:"]
      }), 
      modelId: this.modelId, 
      contentType: "application/json",
      accept: "application/json"
    };
    const response = await this.bedrockClient.send(new InvokeModelWithResponseStreamCommand(input));

    await this.displayNewBotAnswerWithStreaming(video, response)
  }

  async retrieveAnswerForLongVideo(video, systemPrompt){
    const numberOfFragments = parseInt(video.videoScript.length/this.maxCharactersForVideoScript) + 1

    const estimatedTime = (numberOfFragments * 25/60).toFixed(1) // Assuming one LLM call is 25 seconds.
    await this.displayNewBotAnswer(video, `Long video is detected. Estimated answer time is ${estimatedTime} minutes or sooner.`)

    var [currentFragment, rollingAnswer] = [1,""]
    // For every video fragment, try to find the answer to the question, if and not found just write partial answer and go to next iteration.
    while(currentFragment <= numberOfFragments){
      var [videoScriptPrompt, chatPrompt, instructionPrompt, partialAnswerPrompt] = ["","","",""]

      // Construct the prompt containing the video script (the text representation of the current part of the video)
      videoScriptPrompt += "The video timeline has information about the voice heard in the video, visual scenes seen, visual texts visible, and any face or celebrity detected. "
      videoScriptPrompt += "The numbers on the left represents the seconds into the video where the information was extracted.\n"
      videoScriptPrompt += `Because the video is long, the video timeline is split into ${numberOfFragments} parts. `
      if (currentFragment == numberOfFragments){
        videoScriptPrompt += "Below is the last part of the video timeline.\n"
      }else if (currentFragment == 1){
        videoScriptPrompt += "Below is the first part of the video timeline.\n"
      }else{
        videoScriptPrompt += `Below is the part ${currentFragment.toString()} out of ${numberOfFragments} of the video timeline.\n`
      }
      
      // The video script part
      const currentVideoScriptFragment = video.videoScript.substring((currentFragment - 1)*this.maxCharactersForVideoScript, currentFragment*this.maxCharactersForVideoScript)
      videoScriptPrompt += `<VideoTimeline>\n${currentVideoScriptFragment}\n</VideoTimeline>\n\n`
      
      // Add the conversation between user and bot for context
      if (video.chatLength <= this.maxCharactersForChat || video.chatSummary == ""){ 
        chatPrompt += "You had a conversation with a user as below. Your previous answers may be based on the knowledge of the whole video timeline, not only part of it.\n"
        chatPrompt += "<ChatSummary>\n"
        for (const i in video.chats){
          if (video.chats[i].actor == "You"){
            chatPrompt += `\nUser: ${video.chats[i].text}`
          }else{
            chatPrompt += `\nYou: ${video.chats[i].text}`
          }
        }
        chatPrompt += "\n</ChatSummary>\n\n"
      }else{ // If the chat history is too long, so that we include the summary of the chats + last chat only.
        chatPrompt += "You had a conversation with a user as summarized below. Your previous answers may be based on the knowledge of the whole video timeline, not only part of it.\n"
        chatPrompt += `<ChatSummary>\n${video.chatSummary}</ChatSummary>\n`
        chatPrompt += "Now the user comes back with a reply below.\n"
        chatPrompt += `<Question>\n${video.chats[video.chats.length - 1].text}\n</Question>\n\n`
      }

      // Add previous partial answer from previous parts of the videos to gather information so far
      if (rollingAnswer != ""){
        partialAnswerPrompt += "You also saved some partial answer from the previous video parts below. You can use this to provide the final answer or the next partial answer.\n"
        partialAnswerPrompt += `<PreviousPartialAnswer>\n${rollingAnswer}\n</PreviousPartialAnswer>\n\n`
      }

      if (currentFragment == (numberOfFragments + 1)){
        instructionPrompt += "Now, since there is no more video part left, provide your final answer directly without XML tag."
      }else{
        instructionPrompt += "When answering the question, do not use XML tags and do not mention about video timeline, chat summary, or partial answer.\n"
        instructionPrompt += "If you have the final answer already, WRITE |BREAK| at the very end of your answer.\n"
        instructionPrompt += "If you stil need the next part to come with final answer, then write any partial answer based on the current <VideoTimeline>, <ChatSummary>, <Question> if any, and <PreviousPartialAnswer>.\n" 
        instructionPrompt += "This partial answer will be included when we ask you again right after this by giving the video timeline for the next part, so that you can use the current partial answer.\n"
        instructionPrompt += `For question that needs you to see the whole video such as "is there any?", "does it have", "is it mentioned", DO NOT provide final answer until we give you all ${numberOfFragments} parts of the video timeline.\n`
      }
      instructionPrompt += "\n\nAssistant:"

      const prompt = systemPrompt + videoScriptPrompt + chatPrompt + partialAnswerPrompt + instructionPrompt
      const input = { 
        body: JSON.stringify({
          prompt: prompt,
          max_tokens_to_sample: 1000,
          stop_sequences: ["\nUser:"]
        }), 
        modelId: this.modelId, 
        contentType: "application/json",
        accept: "application/json"
      };
      const response = await this.bedrockClient.send(new InvokeModelCommand(input)); 
      rollingAnswer = JSON.parse(new TextDecoder().decode(response.body)).completion

      // When the answer is found without having to go through all video fragments, then break and make it as a final answer.
      if (rollingAnswer.includes("|BREAK|")) {
        rollingAnswer = rollingAnswer.replace("|BREAK|","")
        break
      }

      currentFragment += 1
    }

    await this.displayNewBotAnswer(video, rollingAnswer)
  }

  async handleChat(videoIndex, target){
    var video = this.state.videos[videoIndex]

    if(video.chats.length <= 0) return // Ignore if no chats are there
    if(video.chats[video.chats.length - 1].actor != "You") return // Ignore if last message was from bot

    this.addChatWaitingSpinner(video)

    var systemPrompt = "\n\nHuman: You are an expert in analyzing video and you can answer questions about the video given the video timeline. Answer in the same language as the question from user.\n"
    systemPrompt += "Do not quote the whole video timeline row when answering. Assume the user is not aware of the video timeline. Also if you do not know, say do not know. Do not make up wrong answers.\n\n"
    systemPrompt += "To help you, here is the summary of the whole video that you previously wrote:\n"
    systemPrompt += `${video.summary}\n\n`
    systemPrompt += "Also, here are the entities and sentiment that you previously extracted. Each row is entity|sentiment|sentiment's reason:\n"
    systemPrompt += `${video.rawEntities}\n\n`

    if(video.videoScript.length < this.maxCharactersForSingleLLMCall){
      await this.retrieveAnswerForShortVideo(video, systemPrompt) 
    } else {
      // Long video gets split into fragments for the video script. For each fragment, it goes to LLM to find the partial answer. 
      await this.retrieveAnswerForLongVideo(video, systemPrompt)
    }

    await this.removeChatWaitingSpinner(video)
    this.updateChatSummary(video)

  }

  async displayNewBotAnswer(video, text){
    video.chats.push({actor: "Bot", text: text})
    video.chatLength += text.length
    this.setState({videos: this.state.videos})
  }

  async displayNewBotAnswerWithStreaming(video, response){
    video.chats.push({actor: "Bot", text: ""})
    for await (const event of response.body) {
        if (event.chunk && event.chunk.bytes) {
            const chunk = JSON.parse(Buffer.from(event.chunk.bytes).toString("utf-8"));
            video.chats[video.chats.length - 1].text += chunk.completion
            video.chatLength += chunk.completion.length // Increment the length of the chat
            this.setState({videos: this.state.videos})
        }
    };
  }

  async removeChatWaitingSpinner(video){
    video.chatWaiting = false
    this.setState({videos: this.state.videos})
  }

  async addChatWaitingSpinner(video){
    video.chatWaiting = true
    this.setState({videos: this.state.videos})
  }

  async updateChatSummary(video){
    // If the chats between bot and human is roughly more than 5000 characters (1250 tokens) then summarize the chats, otherwise ignore and return.
    if (video.chatLength < this.maxCharactersForChat) return;

    var prompt = "\n\nHuman: You are an expert in summarizing chat. The chat is between user and a bot. The bot answers questions about a video.\n"

    if (video.chatSummary == ""){
      prompt += "Find the chat below.\n<Chat>\n"
      for (const i in video.chats){
        if (video.chats[i].actor == "You"){
            prompt += `\nUser: ${video.chats[i].text}`
        }else{
          prompt += `\nYou: ${video.chats[i].text}`
        }
      }
      prompt += "\n</Chat>\nSummarize the chat in no longer than 1 page."
    }else{
      prompt += "Below is the summary of the chat so far.\n"
      prompt += `<ChatSummary>\n${video.chatSummary}\n</ChatSummary>\n`
      prompt += "You recently replied to the User as below.\n"
      prompt += `<Reply>\n${video.chats[video.chats.length - 1].text}\n</Reply>\n`
      prompt += "Summarize the whole chat including your newest reply in no longer than 1 page."
    }
    prompt += "\n\nAssistant:"

    const input = { 
      body: JSON.stringify({
        prompt: prompt,
        max_tokens_to_sample: Math.round(this.maxCharactersForChat/this.estimatedCharactersPerToken),
        stop_sequences: ["\nUser:", "\nYou:"]
      }), 
      modelId: this.modelId, 
      contentType: "application/json",
      accept: "application/json"
    };
    const response = await this.bedrockClient.send(new InvokeModelCommand(input));
    video.chatSummary = JSON.parse(new TextDecoder().decode(response.body)).completion
    this.setState({videos: this.state.videos})
  }

  async handleSearchByNameTextChange(event){
    this.setState({searchByNameText: event.target.value })
  }

  async handleSearchByAboutTextChange(event){
    this.setState({searchByAboutText: event.target.value })
  }

  async handleSearchByDateChange(event){
    const startDate = event[0]
    const endDate = event[1]
    this.setState({
      searchByStartDate: startDate,
      searchByEndDate: endDate
    })
  }

  async handleSearch(){
    await this.setState({pages: []})
    var [videos, pages] = await this.fetchVideos(0)
    this.setState({videos: videos, pages: pages})
  }

  renderEntities(video, videoIndex){
    return video.entities.map((entity, i) => {
      return (
        <Row key={`video-${videoIndex}-entity-${i}`}>
          <Col xs={3} className="left-align-no-bottom-margin">{entity[0]}</Col>
          <Col xs={1} className="left-align-no-bottom-margin">{entity[1]}</Col>
          <Col className="left-align-no-bottom-margin">{entity[2]}</Col>
        </Row>
      )
    })
  }
  
  renderVideoTableBody() {
    // Get which page is active to make sure Accordion keeps open for same page and will auto close for page switch.
    var activePage = 0
    Object.keys(this.state.pages).forEach((i) => {if(this.state.pages[i].active) activePage = i})

    // Render the video names as Accordion headers.
    return this.state.videos.map((video, i) => { 
        return (
          <Accordion.Item key={`video-container-${i}`}  eventKey={video.index.toString() + activePage.toString()}>
            <Accordion.Header onClick={this.videoClicked.bind(this, video)}>{video.name}</Accordion.Header>
            <Accordion.Body>
              <Row>
                <Col>
                  { !video.videoShown ? <Button variant="info" size="sm" onClick={this.showVideo.bind(this,video)}>Show video</Button> : <video width="100%" controls><source src={video.url} type="video/mp4" /></video> }
                </Col>
              </Row>
              <Row>
                <Col className={video.loaded  ? "" : "d-none"}>
                  <Row><Col><h5 align="left">Summary:</h5></Col></Row>
                  <Row><Col><p className="paragraph-with-newlines" align="left">{typeof video.summary === "undefined" ? "Summary is not ready yet" : video.summary }</p></Col></Row>
                </Col>
              </Row>
              <Row>
                <Col className={video.loaded  ? "" : "d-none"}>
                  <Row><Col><h5 align="left">Entities:</h5></Col></Row>
                  { typeof video.entities === "undefined" ? <Row><Col><p className="paragraph-with-newlines" align="left">Entity list is not ready yet</p></Col></Row> : <Row><Col className="left-align-no-bottom-margin" xs={3}><b>Entity</b></Col><Col className="left-align-no-bottom-margin" xs={1}><b>Sentiment</b></Col><Col className="left-align-no-bottom-margin"><b>Reason</b></Col></Row> }
                  { typeof video.entities !== "undefined" ? this.renderEntities(video, i) : "" }
                  <br/>
                </Col>
              </Row>
              <Row>
                <Col className={(video.loaded && typeof video.videoScript !== "undefined" ) ? "" : "d-none"}>
                  <Row><Col><h5 align="left">Ask about the video:</h5></Col></Row>
                    {this.renderChat(video)}
                    { video.chatWaiting ? <Row><Col><Spinner className="chat-spinner" size="sm" animation="grow" variant="info" role="status"><span className="visually-hidden">Thinking ...</span></Spinner></Col></Row> : "" }
                  <Row><Col>
                    <Form.Control
                      type="text"
                      id={("video-" + video.index + "-chat-input")}
                      placeholder="e.g. Why is this video funny or interesting?"
                      onKeyDown={this.handleChatInputChange.bind(this, video)}
                      disabled={ video.chatWaiting ? true : false }
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
              <p className="left-align-no-bottom-margin">
                <Form.Label >Search videos with filters</Form.Label>
              </p>
              <InputGroup className="mb-3">
                <Form.Control
                  type="text"
                  placeholder="Video name contains . . ."
                  aria-label="SearchName"
                  onChange={this.handleSearchByNameTextChange.bind(this)}
                  value={this.state.searchByNameText}
                />
                <DatePicker
                  selectsRange={true}
                  startDate={this.state.searchByStartDate}
                  endDate={this.state.searchByEndDate}
                  onChange={(update) => {
                    this.handleSearchByDateChange(update);
                  }}
                  isClearable={true}
                  showIcon={true}
                  placeholderText="Video uploaded between . . ."
                />
             </InputGroup>
             <Form.Control
                type="text"
                placeholder="[semantic search] Video about . . ."
                aria-label="SearchAbout"
                onChange={this.handleSearchByAboutTextChange.bind(this)}
                value={this.state.searchByAboutText}
              />
              <div className="d-grid gap-2 search-button">
                <Button onClick={this.handleSearch.bind(this)} variant="success" id="search-video">
                  Search
                </Button>
              </div>
              
            </Col>
            {/* <Col></Col> */}
          </Row>
          <Row><Col><hr></hr></Col></Row>
          <Row><Col>         
            <Pagination>
              { Object.keys(this.state.pages).map((i) => { return this.renderPagination(i) })}
            </Pagination>
          </Col></Row>
          <Row><Col>
            <Accordion>
              { this.state.pageLoading ? <Spinner className="page-spinner" animation="grow" variant="info" role="status"><span className="visually-hidden">Fetching videos ...</span></Spinner> : '' }
              { Object.keys(this.state.pages).length > 0 && this.state.videos.length > 0 ? this.renderVideoTableBody() : ""}
              { this.state.firstFetch && Object.keys(this.state.pages).length == 0 && this.state.videos.length == 0 ? <p>Found no relevant video in S3 "{this.rawFolder}" folder</p> : ""}
            </Accordion>
          </Col></Row>
        </Col>
      </Row>
    )
  }
}