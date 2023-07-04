import React from 'react'
import Header1 from './Header1'
import Header2 from './Header2'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faXmark, faCheck } from '@fortawesome/free-solid-svg-icons'

const handleChurnPercentage = (churn) => {
    return (!isNaN(churn) && churn >= 0.50)
}
const handleChurnColours = (churn) => {
    return handleChurnPercentage(churn) ? "brilliant-rose" : "old-gold"
}
// const handlePercentAnimation = (churn) => {
//     let percent = document.querySelector("#percentage")
//     let interval = 100000

//     percent.forEach(element => {
//         let start = 0
//         let end = churn | 1
//         let duration = (interval / end).toFixed(2)
//         let counter = setInterval(() => {
//             start+=0.01
//             element.textContent = start.toFixed(2) + "%"
//             if(start == end) {
//                 clearInterval(counter)
//             }
//         }, duration)
//     });
// }

function ChurnPercentage({churn = 0}) {
    return (
        <div className='py-3 bg-dark-green rounded-pill'>
            <Header2 className={"text-white"}> Churn Rate</Header2>
            <Header1 id="percentage" className={"display-1 " + (handleChurnColours(churn))}>
                {
                    isNaN(churn) ? "0.00%": (churn * 100).toFixed(2) + "%"
                }
            </Header1>
            <Header1 className={(handleChurnColours(churn))}>
                {
                    handleChurnPercentage(churn) ?  [<FontAwesomeIcon icon={faCheck} className='brilliant-rose' /> , " Churned!"] : [<FontAwesomeIcon icon={faXmark} className='old-gold' /> , " Not Churned!"]
                }
            </Header1>
        </div>
    )
}

export default ChurnPercentage