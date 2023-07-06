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
                    handleChurnPercentage(churn) ?  [<FontAwesomeIcon icon={faXmark} className='brilliant-rose' /> , " Churned!"] : [<FontAwesomeIcon icon={faCheck} className='old-gold' /> , " Not Churned!"]
                }
            </Header1>
        </div>
    )
}

export default ChurnPercentage