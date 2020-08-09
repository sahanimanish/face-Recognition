
const video = document.getElementById('video')
const container=document.getElementById('container')

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('./models')
]).then(startVideo)



function loadImages() {
  const labels = ['Manish', 'Narendra Modi', 'Obama', 'Akshay Kumar','Tony Stark']
  return Promise.all(
    labels.map(async label => {
      const descriptions = []
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(`./Images/${label}/${i}.jpg`)
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        console.log(detections.descriptor)
        descriptions.push(detections.descriptor)
        
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}


async function loadImageEncodings(){
  const res=await fetch('./Face_Encodings.json')
  let data=await res.json()
  return data.map(d=>{
   let arr=d._descriptors.map(t=>new Float32Array(Object.values(t)))
  
   return new faceapi.LabeledFaceDescriptors(d._label, arr) 
  })

}


function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  )
}


video.addEventListener('play',async () => {
  const canvas = faceapi.createCanvasFromMedia(video)
  container.appendChild
  (canvas)
  
  const displaySize = { width: video.width, height: video.height }
    faceapi.matchDimensions(canvas, displaySize)
    
  const labeledFaceDescriptors=await loadImageEncodings()
 
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
  setInterval(async () => {
    console.log('running')
    const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
      drawBox.draw(canvas)
    })
  }, 100)
})
